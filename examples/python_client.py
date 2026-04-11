#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMini-V1 Python Client SDK Example
=====================================

This module provides a comprehensive example of using OpenAI-compatible API
with the OpenMini-V1 local inference server.

Features:
- Chat completion (streaming & non-streaming)
- Multi-modal support (vision, TTS, STT)
- Embedding generation
- Async client support
- Error handling and retry logic
- Performance benchmarking

Author: OpenMini-V1 Community
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx


# =============================================================================
# Type Definitions (Pydantic-style models for IDE auto-completion)
# =============================================================================


class MessageRole(str, Enum):
    """Chat message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class ChatMessage:
    """
    Chat message with full type hints.

    Attributes:
        role: Message role (system/user/assistant/function)
        content: Message content (string or multi-modal content list)
        name: Optional function name for function messages
    """

    role: Union[MessageRole, str]
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API request."""
        msg: Dict[str, Any] = {"role": self.role.value if isinstance(self.role, MessageRole) else self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class ChatCompletionRequest:
    """
    Chat completion request parameters.

    Attributes:
        model: Model identifier to use
        messages: List of chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        stream: Enable streaming response
        stop: Stop sequences
        presence_penalty: Presence penalty (-2 to 2)
        frequency_penalty: Frequency penalty (-2 to 2)
    """

    model: str = "openmini-7b"
    messages: List[ChatMessage] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API request dictionary."""
        req: Dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        if self.stop:
            req["stop"] = self.stop
        if self.presence_penalty != 0.0:
            req["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            req["frequency_penalty"] = self.frequency_penalty
        return req


@dataclass
class UsageStatistics:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_estimate(self) -> float:
        """Estimate cost based on token usage (simplified)."""
        # Assuming $0.001/1K tokens for estimation
        return (self.total_tokens / 1000) * 0.001


@dataclass
class Choice:
    """Single completion choice."""

    index: int = 0
    message: Optional[ChatMessage] = None
    delta: Optional[Dict[str, Any]] = None  # For streaming
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionResponse:
    """
    Chat completion response.

    Attributes:
        id: Unique response identifier
        object: Object type ("chat.completion")
        created: Unix timestamp
        model: Model used
        choices: List of completion choices
        usage: Token usage statistics
    """

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[UsageStatistics] = None

    @property
    def content(self) -> str:
        """Get the main response content."""
        if self.choices and self.choices[0].message:
            return str(self.choices[0].message.content)
        return ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletionResponse":
        """Create instance from API response dictionary."""
        choices = []
        for choice_data in data.get("choices", []):
            choice = Choice(
                index=choice_data.get("index", 0),
                finish_reason=choice_data.get("finish_reason"),
            )
            if "message" in choice_data:
                msg_data = choice_data["message"]
                choice.message = ChatMessage(
                    role=msg_data.get("role", "assistant"),
                    content=msg_data.get("content", ""),
                )
            if "delta" in choice_data:
                choice.delta = choice_data["delta"]
            choices.append(choice)

        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            usage = UsageStatistics(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
        )


@dataclass
class EmbeddingResponse:
    """Embedding generation response."""

    object: str = "list"
    data: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: Optional[UsageStatistics] = None

    @property
    def embeddings(self) -> List[List[float]]:
        """Extract embedding vectors."""
        return [item.get("embedding", []) for item in self.data]


# =============================================================================
# Client Implementation
# =============================================================================


class OpenMiniClient:
    """
    OpenMini-V1 client with OpenAI-compatible API.

    This client provides a high-level interface for interacting with the
    OpenMini-V1 inference server, supporting both synchronous and asynchronous
    operations.

    Example:
        >>> client = OpenMiniClient(base_url="http://localhost:8080")
        >>> response = client.chat.create(messages=[...])
        >>> print(response.content)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str = "local-dev",
        timeout: float = 120.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """
        Initialize the OpenMini-V1 client.

        Args:
            base_url: Base URL of the OpenMini-V1 server
            api_key: API key (use "local-dev" for local development)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts on failure
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl

        # Initialize HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
            verify=verify_ssl,
        )
        self._async_client: Optional[httpx.AsyncClient] = None

    @property
    def chat(self) -> "_ChatCompletion":
        """Access chat completion interface."""
        return _ChatCompletion(self)

    @property
    def embeddings(self) -> "_Embeddings":
        """Access embeddings interface."""
        return _Embeddings(self)

    def health_check(self) -> bool:
        """
        Check server health status.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.

        Returns:
            List of available models with their metadata
        """
        response = self._client.get("/v1/models")
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()

    def __enter__(self) -> "OpenMiniClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> "OpenMiniClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.aclose()


class _ChatCompletion:
    """Chat completion interface."""

    def __init__(self, client: OpenMiniClient):
        self._client = client

    def create(
        self,
        *,
        model: str = "openmini-7b",
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Union[ChatCompletionResponse, Iterator[Dict[str, Any]]]:
        """
        Create a chat completion.

        Args:
            model: Model identifier
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stream: Enable streaming mode
            stop: Stop sequences

        Returns:
            ChatCompletionResponse or streaming iterator
        """
        # Normalize messages
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized_messages.append(msg.to_dict())
            elif isinstance(msg, dict):
                normalized_messages.append(msg)
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

        payload = {
            "model": model,
            "messages": normalized_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stop:
            payload["stop"] = stop

        # Add extra kwargs
        payload.update(kwargs)

        if stream:
            return self._stream_request("/v1/chat/completions", payload)
        else:
            return self._sync_request("/v1/chat/completions", payload)

    def _sync_request(self, endpoint: str, payload: Dict[str, Any]) -> ChatCompletionResponse:
        """Make synchronous request."""
        for attempt in range(self._client.max_retries):
            try:
                response = self._client._client.post(endpoint, json=payload)
                response.raise_for_status()
                return ChatCompletionResponse.from_dict(response.json())
            except httpx.HTTPStatusError as e:
                if attempt == self._client.max_retries - 1:
                    raise RuntimeError(f"API request failed after {self._client.max_retries} attempts: {e}")
                time.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self._client.max_retries - 1:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Unexpected error in request loop")

    def _stream_request(self, endpoint: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Make streaming request."""
        try:
            with self._client._client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        import json

                        yield json.loads(data_str)
        except Exception as e:
            raise RuntimeError(f"Streaming request failed: {e}")


class _Embeddings:
    """Embeddings interface."""

    def __init__(self, client: OpenMiniClient):
        self._client = client

    def create(
        self,
        *,
        model: str = "openmini-embeddings",
        input: Union[str, List[str]],
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Create embeddings for text input(s).

        Args:
            model: Embedding model identifier
            input: Text string or list of texts

        Returns:
            EmbeddingResponse containing embedding vectors
        """
        payload = {
            "model": model,
            "input": input,
        }
        payload.update(kwargs)

        response = self._client._client.post("/v1/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()

        return EmbeddingResponse(
            object=data.get("object", "list"),
            data=data.get("data", []),
            model=data.get("model", model),
            usage=(
                UsageStatistics(**data["usage"])
                if "usage" in data
                else None
            ),
        )


# =============================================================================
# Async Client
# =============================================================================


class AsyncOpenMiniClient:
    """
    Asynchronous OpenMini-V1 client for high-concurrency applications.

    Example:
        >>> async with AsyncOpenMiniClient() as client:
        ...     response = await client.chat.create(messages=[...])
        ...     print(response.content)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str = "local-dev",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization of async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    @property
    def chat(self) -> "_AsyncChatCompletion":
        """Access async chat completion interface."""
        return _AsyncChatCompletion(self)

    async def health_check(self) -> bool:
        """Check server health asynchronously."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the async client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncOpenMiniClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class _AsyncChatCompletion:
    """Async chat completion interface."""

    def __init__(self, client: AsyncOpenMiniClient):
        self._client = client

    async def create(
        self,
        *,
        model: str = "openmini-7b",
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletionResponse, AsyncIterator[Dict[str, Any]]]:
        """Create async chat completion."""
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized_messages.append(msg.to_dict())
            elif isinstance(msg, dict):
                normalized_messages.append(msg)
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

        payload = {
            "model": model,
            "messages": normalized_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        payload.update(kwargs)

        if stream:
            return self._stream_request(payload)
        else:
            return await self._sync_request(payload)

    async def _sync_request(self, payload: Dict[str, Any]) -> ChatCompletionResponse:
        """Make synchronous async request."""
        client = await self._client._get_client()
        for attempt in range(self._client.max_retries):
            try:
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                return ChatCompletionResponse.from_dict(response.json())
            except httpx.HTTPStatusError as e:
                if attempt == self._client.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)

        raise RuntimeError("Max retries exceeded")

    async def _stream_request(self, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Make streaming async request."""
        client = await self._client._get_client()
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    import json

                    yield json.loads(data_str)


# =============================================================================
# Utility Functions
# =============================================================================


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string for vision API.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded image string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """
    Get MIME type for image file.

    Args:
        image_path: Path to image file

    Returns:
        MIME type string (e.g., "image/png")
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/png")


# =============================================================================
# Examples and Demo Functions
# =============================================================================


def example_basic_chat_completion():
    """
    Example 1: Basic Chat Completion (非流式)

    Demonstrates simple question-answer interaction with the model.
    """
    print("=" * 60)
    print("Example 1: Basic Chat Completion")
    print("=" * 60)

    with OpenMiniClient(base_url="http://localhost:8080") as client:
        # Check server health
        if not client.health_check():
            print("❌ Server is not running. Please start OpenMini-V1 first.")
            return

        print("✅ Server is healthy")

        # Create chat completion request
        request = ChatCompletionRequest(
            model="openmini-7b",
            messages=[
                ChatMessage(role="system", content="你是一个有帮助的 AI 助手。"),
                ChatMessage(role="user", content="请用三句话介绍 Go 语言的特点。"),
            ],
            max_tokens=512,
            temperature=0.7,
        )

        try:
            response = client.chat.create(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            print(f"\n📝 Response:")
            print(f"{response.content}")

            if response.usage:
                print(f"\n📊 Token Usage:")
                print(f"   Prompt tokens: {response.usage.prompt_tokens}")
                print(f"   Completion tokens: {response.usage.completion_tokens}")
                print(f"   Total tokens: {response.usage.total_tokens}")
                print(f"   Estimated cost: ${response.usage.cost_estimate:.4f}")

        except Exception as e:
            print(f"❌ Error: {e}")


def example_streaming_chat():
    """
    Example 2: Streaming Chat Completion (流式输出)

    Demonstrates real-time token-by-token output.
    """
    print("\n" + "=" * 60)
    print("Example 2: Streaming Chat Completion")
    print("=" * 60)

    with OpenMiniClient(base_url="http://localhost:8080") as client:
        if not client.health_check():
            print("❌ Server is not running.")
            return

        print("✅ Streaming response:\n")

        messages = [
            ChatMessage(role="user", content="写一首关于春天的诗，要求五言绝句。"),
        ]

        try:
            stream = client.chat.create(
                model="openmini-7b",
                messages=messages,
                max_tokens=256,
                temperature=0.9,
                stream=True,
            )

            full_content = []
            for chunk in stream:
                if chunk.get("choices"):
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_content.append(content)

            print("\n\n✅ Streaming completed")

        except Exception as e:
            print(f"❌ Error: {e}")


def example_multi_turn_conversation():
    """
    Example 3: Multi-turn Conversation (多轮对话)

    Demonstrates maintaining conversation context across multiple turns.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multi-turn Conversation")
    print("=" * 60)

    with OpenMiniClient(base_url="http://localhost:8080") as client:
        if not client.health_check():
            print("❌ Server is not running.")
            return

        # Maintain conversation history
        conversation_history: List[ChatMessage] = [
            ChatMessage(role="系统", content="你是一个专业的技术顾问，擅长解答编程问题。"),
        ]

        questions = [
            "什么是 RESTful API？",
            "能给我一个具体的例子吗？",
            "Go 语言中如何实现？",
        ]

        try:
            for i, question in enumerate(questions, 1):
                print(f"\n👤 User (Turn {i}): {question}")

                # Add user message to history
                conversation_history.append(ChatMessage(role="user", content=question))

                # Get assistant response
                response = client.chat.create(
                    model="openmini-7b",
                    messages=conversation_history,
                    max_tokens=384,
                    temperature=0.7,
                )

                print(f"\n🤖 Assistant: {response.content}\n")

                # Add assistant response to history
                conversation_history.append(
                    ChatMessage(role="assistant", content=response.content)
                )

            print(f"\n✅ Total conversation turns: {len(questions)}")

        except Exception as e:
            print(f"❌ Error: {e}")


def example_vision_understanding(image_path: Optional[str] = None):
    """
    Example 4: Image Understanding (视觉理解)

    Demonstrates multi-modal capabilities with image input.
    """
    print("\n" + "=" * 60)
    print("Example 4: Image Understanding (Vision)")
    print("=" * 60)

    # Use a sample image path or create a placeholder message
    if not image_path:
        print("ℹ️  No image path provided. Showing example structure.")
        print("To test with actual image, provide image_path parameter.\n")

        # Show example structure
        example_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片的内容"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{get_image_media_type('example.png')};base64,{encode_image_to_base64('example.png')}"
                    },
                },
            ],
        }

        print("📋 Example message structure:")
        import json

        print(json.dumps(example_message, indent=2, ensure_ascii=False))
        return

    with OpenMiniClient(base_url="http://localhost:8080") as client:
        if not client.health_check():
            print("❌ Server is not running.")
            return

        try:
            # Encode image
            image_base64 = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)

            # Create vision request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请详细描述这张图片的内容"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}"
                            },
                        },
                    ],
                }
            ]

            response = client.chat.create(
                model="openmini-vision-7b",  # Vision-capable model
                messages=messages,
                max_tokens=512,
            )

            print(f"\n🖼️  Image Analysis Result:")
            print(f"{response.content}")

        except FileNotFoundError:
            print(f"❌ Image file not found: {image_path}")
        except Exception as e:
            print(f"❌ Error: {e}")


def example_embedding_generation():
    """
    Example 5: Text Embedding Generation (文本向量化)

    Demonstrates generating embeddings for semantic search and similarity.
    """
    print("\n" + "=" * 60)
    print("Example 5: Text Embedding Generation")
    print("=" * 60)

    with OpenMiniClient(base_url="http://localhost:8080") as client:
        if not client.health_check():
            print("❌ Server is not running.")
            return

        texts = [
            "人工智能正在改变世界",
            "机器学习是AI的一个子集",
            "今天天气真好",
        ]

        try:
            response = client.embeddings.create(
                model="openmini-embeddings",
                input=texts,
            )

            print(f"\n📊 Embedding Results:")
            print(f"Model: {response.model}")
            print(f"Number of embeddings: {len(response.embeddings)}")

            if response.embeddings:
                print(f"\nFirst embedding dimension: {len(response.embeddings[0])}")
                print(f"First 10 values: {response.embeddings[0][:10]}")

            # Calculate similarity between first two texts
            if len(response.embeddings) >= 2:
                import numpy as np

                vec1 = np.array(response.embeddings[0])
                vec2 = np.array(response.embeddings[1])
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                print(f"\n🔗 Similarity (Text 1 vs Text 2): {similarity:.4f}")

            if response.usage:
                print(f"\n📈 Usage: {response.usage.total_tokens} tokens")

        except ImportError:
            print("⚠️  Install numpy for similarity calculation: pip install numpy")
        except Exception as e:
            print(f"❌ Error: {e}")


def example_async_client():
    """
    Example 6: Async Client Usage (异步客户端)

    Demonstrates using the async client for concurrent requests.
    """
    print("\n" + "=" * 60)
    print("Example 6: Async Client (Concurrent Requests)")
    print("=" * 60)

    async def run_async_example():
        async with AsyncOpenMiniClient(base_url="http://localhost:8080") as client:
            if not await client.health_check():
                print("❌ Server is not running.")
                return

            print("✅ Async client connected\n")

            # Prepare multiple requests
            tasks = []
            questions = [
                "什么是微服务架构？",
                "Docker 和 Kubernetes 的区别？",
                "解释一下 CI/CD 流程",
            ]

            for q in questions:
                task = client.chat.create(
                    model="openmini-7b",
                    messages=[ChatMessage(role="user", content=q)],
                    max_tokens=256,
                )
                tasks.append((q, task))

            # Execute concurrently
            print("⏳ Sending concurrent requests...\n")
            results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

            for (question, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"❌ Question: {question}")
                    print(f"   Error: {result}\n")
                else:
                    print(f"❓ Question: {question}")
                    print(f"💬 Answer: {result.content[:100]}...\n")

    try:
        asyncio.run(run_async_example())
    except Exception as e:
        print(f"❌ Async error: {e}")


def example_error_handling_and_retry():
    """
    Example 7: Error Handling and Retry Logic (错误处理与重试)

    Demonstrates robust error handling patterns.
    """
    print("\n" + "=" * 60)
    print("Example 7: Error Handling & Retry Logic")
    print("=" * 60)

    # Configure client with aggressive retry
    client = OpenMiniClient(
        base_url="http://localhost:8080",
        api_key="local-dev",
        timeout=30.0,
        max_retries=3,
    )

    try:
        with client:
            if not client.health_check():
                print("⚠️  Server unavailable - demonstrating error handling")
                # Simulate error scenarios
                error_scenarios = [
                    ("Invalid model", "nonexistent-model"),
                    ("Empty messages", []),
                    ("Very long prompt", ["x" * 100000]),
                ]

                for scenario_name, scenario_value in error_scenarios:
                    print(f"\n🧪 Testing: {scenario_name}")
                    try:
                        if scenario_name == "Invalid model":
                            response = client.chat.create(
                                model=scenario_value,
                                messages=[
                                    ChatMessage(role="user", content="test")
                                ],
                            )
                        elif scenario_name == "Empty messages":
                            response = client.chat.create(
                                model="openmini-7b",
                                messages=scenario_value,
                            )
                        elif scenario_name == "Very long prompt":
                            response = client.chat.create(
                                model="openmini-7b",
                                messages=[
                                    ChatMessage(
                                        role="user", content=scenario_value
                                    )
                                ],
                            )
                        print(f"   Response: {response.content[:50]}...")
                    except Exception as e:
                        print(f"   ✅ Caught expected error: {type(e).__name__}")
                        print(f"      Message: {str(e)[:100]}")
            else:
                print("✅ Server healthy - testing normal operation")
                response = client.chat.create(
                    model="openmini-7b",
                    messages=[
                        ChatMessage(role="user", content="Say 'Hello'")
                    ],
                    max_tokens=32,
                )
                print(f"✅ Response: {response.content}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def example_performance_benchmark():
    """
    Example 8: Performance Benchmarking (性能基准测试)

    Measures latency and throughput of the API.
    """
    print("\n" + "=" * 60)
    print("Example 8: Performance Benchmarking")
    print("=" * 60)

    client = OpenMiniClient(
        base_url="http://localhost:8080",
        api_key="local-dev",
        timeout=300.0,  # Longer timeout for benchmarking
    )

    if not client.health_check():
        print("❌ Server is not running. Cannot run benchmarks.")
        return

    # Test configuration
    num_requests = 5
    prompt = "用一句话解释量子计算的基本原理。"

    latencies = []
    token_counts = []

    print(f"\n📊 Running {num_requests} requests for benchmarking...\n")

    try:
        with client:
            for i in range(num_requests):
                start_time = time.time()

                response = client.chat.create(
                    model="openmini-7b",
                    messages=[ChatMessage(role="user", content=prompt)],
                    max_tokens=128,
                    temperature=0.7,
                )

                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)

                if response.usage:
                    token_counts.append(response.usage.completion_tokens)

                print(f"   Request {i+1}: {latency:.2f}s | Tokens: {response.usage.completion_tokens if response.usage else 'N/A'}")

        # Calculate statistics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            print("\n📈 Benchmark Results:")
            print(f"   Total requests: {num_requests}")
            print(f"   Average latency: {avg_latency:.2f}s")
            print(f"   Min latency: {min_latency:.2f}s")
            print(f"   Max latency: {max_latency:.2f}s")

            if token_counts:
                avg_tokens = sum(token_counts) / len(token_counts)
                throughput = avg_tokens / avg_latency if avg_latency > 0 else 0
                print(f"   Average tokens: {avg_tokens:.1f}")
                print(f"   Throughput: {throughput:.1f} tokens/s")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def example_list_models():
    """
    Example 9: List Available Models (列出可用模型)

    Demonstrates querying available models from the server.
    """
    print("\n" + "=" * 60)
    print("Example 9: List Available Models")
    print("=" * 60)

    with OpenMiniClient(base_url="http://localhost:8080") as client:
        try:
            models = client.list_models()

            print(f"\n📋 Available Models ({len(models)}):\n")

            for model in models:
                model_id = model.get("id", "unknown")
                owned_by = model.get("owned_by", "unknown")
                print(f"   📦 {model_id:<30} | Owner: {owned_by}")

            if not models:
                print("   ℹ️  No models found. Make sure models are loaded.")

        except Exception as e:
            print(f"❌ Error listing models: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """
    Main function to run all examples.

    Run individual examples by passing example number as argument:
        python python_client.py 1  # Run only example 1
        python python_client.py    # Run all examples
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         OpenMini-V1 Python Client SDK Examples           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Parse command line arguments
    example_num = None
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid argument: {sys.argv[1]}")
            print("Usage: python python_client.py [example_number]")
            sys.exit(1)

    # Map example numbers to functions
    examples = {
        1: ("Basic Chat Completion", example_basic_chat_completion),
        2: ("Streaming Chat Completion", example_streaming_chat),
        3: ("Multi-turn Conversation", example_multi_turn_conversation),
        4: ("Image Understanding (Vision)", lambda: example_vision_understanding(None)),
        5: ("Text Embedding Generation", example_embedding_generation),
        6: ("Async Client Usage", example_async_client),
        7: ("Error Handling & Retry", example_error_handling_and_retry),
        8: ("Performance Benchmarking", example_performance_benchmark),
        9: ("List Available Models", example_list_models),
    }

    if example_num:
        # Run specific example
        if example_num in examples:
            print(f"> Running Example {example_num}: {examples[example_num][0]}\n")
            examples[example_num][1]()
        else:
            print(f"❌ Example {example_num} not found.")
            print(f"\nAvailable examples:")
            for num, (name, _) in examples.items():
                print(f"  {num}. {name}")
    else:
        # Run all examples sequentially
        print("> Running all examples...\n")

        for num, (name, func) in examples.items():
            print(f"\n{'='*60}")
            print(f"[{num}/{len(examples)}] {name}")
            print('='*60)
            try:
                func()
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Example {num} failed: {e}")
                continue

        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()


# =============================================================================
# Installation Instructions (shown when imported as module)
# =============================================================================

"""
Installation Requirements
=========================

Install required packages:

    pip install openai httpx aiohttp numpy pytest mypy ruff

Or use requirements.txt:

    openai>=1.0.0
    httpx>=0.25.0
    aiohttp>=3.9.0
    numpy>=1.24.0
    pytest>=7.4.0
    mypy>=1.5.0
    ruff>=0.1.0

Environment Variables (Optional)
================================

OPENMINI_BASE_URL     - Server URL (default: http://localhost:8080)
OPENMINI_API_KEY      - API key (default: local-dev)
OPENMINI_TIMEOUT      - Request timeout in seconds (default: 120)
OPENMINI_MAX_RETRIES  - Max retry attempts (default: 3)

Connection Troubleshooting
==========================

1. Server not starting?
   - Check if port 8080 is already in use: lsof -i :8088
   - Verify GPU drivers are installed: nvidia-smi
   - Check Rust/CUDA versions meet requirements

2. Connection refused?
   - Ensure server is running: curl http://localhost:8080/health
   - Check firewall settings
   - Verify URL is correct (no trailing slash)

3. Timeout errors?
   - Increase timeout value for large models
   - Check GPU memory usage: watch -n 1 nvidia-smi
   - Reduce max_tokens for faster responses

4. Import errors?
   - Activate virtual environment: source venv/bin/activate
   - Upgrade packages: pip install --upgrade -r requirements.txt
   - Check Python version: python --version (need 3.10+)
"""
