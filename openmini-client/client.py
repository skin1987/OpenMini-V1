"""
OpenMini Python gRPC Client - OpenMini Python gRPC 客户端模块

该模块提供了与 OpenMini Rust 推理服务通信的底层 gRPC 客户端实现。
封装了 gRPC 协议细节，提供简洁的 Python API。

主要组件:
    - OpenMiniClient: 主要客户端类，提供对话和图像理解接口
    - ConnectionPool: gRPC 连接池，管理连接复用
    - Message: 消息数据类
    - ChatResponse: 响应数据类
    - UsageInfo: Token 使用统计类

功能特性:
    - 连接池管理: 自动管理 gRPC 连接，提高性能
    - 流式输出: 支持流式响应，实时获取生成内容
    - 图像理解: 支持图像问答功能
    - 健康检查: 支持服务健康状态检查

使用示例:
    # 创建客户端
    client = OpenMiniClient("localhost", 50051)
    
    # 文本对话
    messages = [Message(role="user", content="你好")]
    for response in client.chat(messages):
        print(response.token, end="")
    
    # 图像理解
    with open("image.jpg", "rb") as f:
        for response in client.image_understanding(f.read(), "描述这张图片"):
            print(response.token, end="")

依赖:
    - grpcio: gRPC Python 库
    - openmini_pb2: 生成的 protobuf 模块
    - openmini_pb2_grpc: 生成的 gRPC 模块

作者: OpenMini Team
版本: 0.1.0
"""

import grpc
from typing import Iterator, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from queue import Queue

# 导入生成的 protobuf 模块
# 这些模块由 protoc 编译器从 openmini.proto 生成
import openmini_pb2
import openmini_pb2_grpc


@dataclass
class Message:
    """
    消息数据类
    
    表示对话中的一条消息，可以是用户消息或助手消息。
    
    Attributes:
        role (str): 消息角色，"user" 或 "assistant"
        content (str): 消息文本内容
        image_data (Optional[bytes]): 图像二进制数据（可选）
    
    Example:
        >>> msg = Message(role="user", content="你好")
        >>> msg_with_image = Message(
        ...     role="user",
        ...     content="描述这张图片",
        ...     image_data=image_bytes
        ... )
    """
    role: str
    content: str
    image_data: Optional[bytes] = None


@dataclass
class UsageInfo:
    """
    Token 使用统计信息
    
    记录一次请求的 token 使用情况，用于监控和计费。
    
    Attributes:
        prompt_tokens (int): 输入 prompt 的 token 数
        completion_tokens (int): 生成的 token 数
        total_tokens (int): 总 token 数
    
    Example:
        >>> usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        >>> print(f"使用了 {usage.total_tokens} 个 token")
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    """
    对话响应数据类
    
    表示流式响应中的单个响应单元。
    
    Attributes:
        session_id (str): 会话 ID，用于保持对话上下文
        token (str): 生成的单个 token（可能是部分词）
        finished (bool): 是否为最后一个响应
        usage (Optional[UsageInfo]): Token 使用统计（仅在最后一个响应中包含）
    
    Example:
        >>> for response in client.chat(messages):
        ...     print(response.token, end="")
        ...     if response.finished:
        ...         print(f"\\n总 token: {response.usage.total_tokens}")
    """
    session_id: str
    token: str
    finished: bool
    usage: Optional[UsageInfo] = None


class ConnectionPool:
    """
    gRPC 连接池
    
    管理 gRPC channel 的创建和复用，提高连接效率。
    使用线程安全的队列实现连接池。
    
    Attributes:
        address (str): 服务地址，格式为 "host:port"
        pool_size (int): 连接池大小
    
    工作原理:
        1. 首次请求时创建新连接
        2. 使用完毕后将连接放回池中
        3. 后续请求优先复用池中的连接
        4. 池满时丢弃多余连接
    
    Example:
        >>> pool = ConnectionPool("localhost", 50051, pool_size=10)
        >>> with pool.channel() as channel:
        ...     stub = openmini_pb2_grpc.OpenMiniStub(channel)
        ...     # 使用 stub 进行调用
    """
    
    def __init__(self, host: str, port: int, pool_size: int = 10):
        """
        初始化连接池
        
        Args:
            host (str): 服务主机地址
            port (int): 服务端口
            pool_size (int): 连接池大小，默认为 10
        """
        self.address = f"{host}:{port}"
        self.pool_size = pool_size
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 1  # 已创建的连接数
        
    def _create_channel(self) -> grpc.Channel:
        """
        创建新的 gRPC channel
        
        Returns:
            grpc.Channel: 新创建的 gRPC channel
        """
        return grpc.insecure_channel(self.address)
    
    def acquire(self) -> grpc.Channel:
        """
        获取一个 gRPC channel
        
        优先从池中获取空闲连接，如果没有则创建新连接。
        
        Returns:
            grpc.Channel: 可用的 gRPC channel
        """
        try:
            # 尝试从池中获取空闲连接
            return self._pool.get_nowait()
        except:  # noqa: E722
            # 池中没有可用连接，创建新连接
            with self._lock:
                if self._created < self.pool_size:
                    self._created += 1
                    return self._create_channel()
            # 超过池大小限制，仍然创建（但不放回池中）
            return self._create_channel()
    
    def release(self, channel: grpc.Channel):
        """
        释放 gRPC channel 回连接池
        
        Args:
            channel (grpc.Channel): 要释放的 channel
        """
        try:
            self._pool.put_nowait(channel)
        except:  # noqa: E722
            # 池已满，丢弃连接
            pass
    
    @contextmanager
    def channel(self):
        """
        上下文管理器，自动获取和释放连接
        
        使用 with 语句确保连接正确释放。
        
        Yields:
            grpc.Channel: 可用的 gRPC channel
        
        Example:
            >>> with pool.channel() as channel:
            ...     stub = openmini_pb2_grpc.OpenMiniStub(channel)
            ...     response = stub.HealthCheck(request)
        """
        ch = self.acquire()
        try:
            yield ch
        finally:
            self.release(ch)


class OpenMiniClient:
    """
    OpenMini gRPC 客户端
    
    提供与 OpenMini 推理服务交互的主要接口。
    支持文本对话、图像理解和健康检查功能。
    
    Attributes:
        pool (ConnectionPool): 连接池实例
        host (str): 服务主机地址
        port (int): 服务端口
    
    Example:
        >>> client = OpenMiniClient("localhost", 50051)
        >>> 
        >>> # 检查服务健康状态
        >>> if client.health_check():
        ...     print("服务正常")
        >>> 
        >>> # 文本对话
        >>> messages = [Message(role="user", content="你好")]
        >>> for response in client.chat(messages):
        ...     print(response.token, end="")
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051, pool_size: int = 10):
        """
        初始化 OpenMini 客户端
        
        Args:
            host (str): 服务主机地址，默认为 "localhost"
            port (int): 服务端口，默认为 50051
            pool_size (int): 连接池大小，默认为 10
        """
        self.pool = ConnectionPool(host, port, pool_size)
        self.host = host
        self.port = port
    
    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Iterator[ChatResponse]:
        """
        对话接口（流式）
        
        发送消息到服务并流式获取响应。
        
        Args:
            messages (List[Message]): 消息列表
            max_tokens (int): 最大生成 token 数，默认 1024
            temperature (float): 温度参数，控制随机性，默认 0.7
                - 0.0: 确定性输出
                - 0.7: 平衡创造性和一致性
                - 1.0+: 更有创造性
            stream (bool): 是否流式输出，默认 True
            
        Yields:
            ChatResponse: 流式响应，每个响应包含一个 token
            
        Example:
            >>> for response in client.chat(messages, max_tokens=512):
            ...     print(response.token, end="")
            ...     if response.finished:
            ...         print(f"\\n使用了 {response.usage.total_tokens} tokens")
        """
        with self.pool.channel() as channel:
            stub = openmini_pb2_grpc.OpenMiniStub(channel)
            
            # 构造 gRPC 请求
            request = openmini_pb2.ChatRequest(
                session_id="",
                messages=[
                    openmini_pb2.Message(
                        role=m.role,
                        content=m.content,
                        image_data=m.image_data or b""
                    )
                    for m in messages
                ],
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # 流式接收响应
            for response in stub.Chat(iter([request])):
                # 解析 usage 信息
                usage = None
                if response.HasField("usage"):
                    usage = UsageInfo(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                
                yield ChatResponse(
                    session_id=response.session_id,
                    token=response.token,
                    finished=response.finished,
                    usage=usage,
                )
    
    def chat_sync(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        同步对话接口
        
        发送消息并获取完整响应文本。
        
        Args:
            messages (List[Message]): 消息列表
            max_tokens (int): 最大生成 token 数
            temperature (float): 温度参数
            
        Returns:
            str: 完整的响应文本
            
        Example:
            >>> response = client.chat_sync([Message("user", "你好")])
            >>> print(response)
            你好！有什么可以帮助你的？
        """
        result = []
        for response in self.chat(messages, max_tokens, temperature, stream=True):
            result.append(response.token)
            if response.finished:
                break
        return "".join(result)
    
    def image_understanding(
        self,
        image_data: bytes,
        question: str,
        stream: bool = True,
    ) -> Iterator[ChatResponse]:
        """
        图像理解接口（流式）
        
        发送图像和问题，获取图像理解响应。
        
        Args:
            image_data (bytes): 图像二进制数据
            question (str): 关于图像的问题
            stream (bool): 是否流式输出，默认 True
            
        Yields:
            ChatResponse: 流式响应
            
        Example:
            >>> with open("photo.jpg", "rb") as f:
            ...     for response in client.image_understanding(f.read(), "描述这张图片"):
            ...         print(response.token, end="")
        """
        with self.pool.channel() as channel:
            stub = openmini_pb2_grpc.OpenMiniStub(channel)
            
            # 构造图像请求
            request = openmini_pb2.ImageRequest(
                session_id="",
                image_data=image_data,
                question=question,
                stream=stream,
            )
            
            if stream:
                # 流式响应
                for response in stub.ImageUnderstandingStream(request):
                    yield ChatResponse(
                        session_id=response.session_id,
                        token=response.token,
                        finished=response.finished,
                    )
            else:
                # 单次响应
                response = stub.ImageUnderstanding(request)
                yield ChatResponse(
                    session_id=response.session_id,
                    token=response.token,
                    finished=response.finished,
                )
    
    def image_understanding_sync(
        self,
        image_data: bytes,
        question: str,
    ) -> str:
        """
        同步图像理解接口
        
        发送图像和问题，获取完整响应文本。
        
        Args:
            image_data (bytes): 图像二进制数据
            question (str): 关于图像的问题
            
        Returns:
            str: 完整的响应文本
            
        Example:
            >>> with open("photo.jpg", "rb") as f:
            ...     answer = client.image_understanding_sync(f.read(), "图片里有什么？")
            ...     print(answer)
        """
        result = []
        for response in self.image_understanding(image_data, question, stream=True):
            result.append(response.token)
            if response.finished:
                break
        return "".join(result)
    
    def health_check(self) -> bool:
        """
        健康检查
        
        检查服务是否正常运行。
        
        Returns:
            bool: True 表示服务健康，False 表示服务不可用
            
        Example:
            >>> if client.health_check():
            ...     print("服务正常")
            ... else:
            ...     print("服务不可用")
        """
        try:
            with self.pool.channel() as channel:
                stub = openmini_pb2_grpc.OpenMiniStub(channel)
                response = stub.HealthCheck(openmini_pb2.HealthRequest())
                return response.healthy
        except Exception:
            return False


def create_client(host: str = "localhost", port: int = 50051) -> OpenMiniClient:
    """
    创建 OpenMini 客户端实例的工厂函数
    
    提供便捷的客户端创建方式。
    
    Args:
        host (str): 服务主机地址，默认为 "localhost"
        port (int): 服务端口，默认为 50051
        
    Returns:
        OpenMiniClient: 客户端实例
        
    Example:
        >>> client = create_client()
        >>> # 或指定地址
        >>> client = create_client("192.168.1.100", 50051)
    """
    return OpenMiniClient(host, port)


def example_chat():
    """
    聊天示例
    
    演示基本的文本对话功能，包括单轮对话和多轮对话。
    """
    print("=" * 50)
    print("示例 1: 基本聊天")
    print("=" * 50)
    
    client = OpenMiniClient("localhost", 50051)
    
    if not client.health_check():
        print("错误: 服务不可用，请先启动 OpenMini 服务")
        return
    
    messages = [
        Message(role="user", content="你好，请用一句话介绍自己")
    ]
    
    print("用户: 你好，请用一句话介绍自己")
    print("助手: ", end="", flush=True)
    
    for response in client.chat(messages, max_tokens=256, temperature=0.7):
        print(response.token, end="", flush=True)
        if response.finished and response.usage:
            print(f"\n[Token 使用: {response.usage.total_tokens}]")
    
    print()
    
    print("=" * 50)
    print("示例 2: 多轮对话")
    print("=" * 50)
    
    session_messages = [
        Message(role="user", content="我叫张三"),
        Message(role="assistant", content="你好张三，很高兴认识你！有什么我可以帮助你的吗？"),
        Message(role="user", content="你还记得我的名字吗？")
    ]
    
    print("用户: 我叫张三")
    print("助手: 你好张三，很高兴认识你！有什么我可以帮助你的吗？")
    print("用户: 你还记得我的名字吗？")
    print("助手: ", end="", flush=True)
    
    for response in client.chat(session_messages, max_tokens=256):
        print(response.token, end="", flush=True)
    print("\n")


def example_image_understanding():
    """
    图像理解示例
    
    演示如何使用图像理解功能分析图片内容。
    """
    print("=" * 50)
    print("示例 3: 图像理解")
    print("=" * 50)
    
    client = OpenMiniClient("localhost", 50051)
    
    if not client.health_check():
        print("错误: 服务不可用")
        return
    
    image_path = "test_image.jpg"
    
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
    except FileNotFoundError:
        print(f"提示: 未找到测试图片 {image_path}")
        print("请将图片放置在当前目录下，或修改 image_path 变量")
        return
    
    print(f"正在分析图片: {image_path}")
    print("问题: 请描述这张图片的内容")
    print("回答: ", end="", flush=True)
    
    for response in client.image_understanding(image_data, "请描述这张图片的内容"):
        print(response.token, end="", flush=True)
    print("\n")


def example_multimodal_chat():
    """
    多模态聊天示例
    
    演示如何在对话中包含图像内容。
    """
    print("=" * 50)
    print("示例 4: 多模态对话")
    print("=" * 50)
    
    client = OpenMiniClient("localhost", 50051)
    
    if not client.health_check():
        print("错误: 服务不可用")
        return
    
    image_path = "test_image.jpg"
    
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
    except FileNotFoundError:
        print(f"提示: 未找到测试图片 {image_path}")
        return
    
    messages = [
        Message(
            role="user",
            content="这张图片里有什么？请详细描述。",
            image_data=image_data
        )
    ]
    
    print("用户: [发送了一张图片] 这张图片里有什么？")
    print("助手: ", end="", flush=True)
    
    for response in client.chat(messages, max_tokens=512):
        print(response.token, end="", flush=True)
    print("\n")


def example_sync_chat():
    """
    同步聊天示例
    
    演示使用同步方式获取完整响应。
    """
    print("=" * 50)
    print("示例 5: 同步聊天")
    print("=" * 50)
    
    client = OpenMiniClient("localhost", 50051)
    
    if not client.health_check():
        print("错误: 服务不可用")
        return
    
    messages = [Message(role="user", content="请写一首关于春天的短诗")]
    
    print("用户: 请写一首关于春天的短诗")
    print("助手: ", end="")
    
    response = client.chat_sync(messages, max_tokens=256)
    print(response)
    print()


def example_connection_pool():
    """
    连接池使用示例
    
    演示如何配置和使用连接池提高性能。
    """
    print("=" * 50)
    print("示例 6: 连接池配置")
    print("=" * 50)
    
    client = OpenMiniClient(
        host="localhost",
        port=50051,
        pool_size=20
    )
    
    if not client.health_check():
        print("错误: 服务不可用")
        return
    
    prompts = [
        "什么是人工智能？",
        "什么是机器学习？",
        "什么是深度学习？"
    ]
    
    for prompt in prompts:
        messages = [Message(role="user", content=prompt)]
        print(f"用户: {prompt}")
        print("助手: ", end="")
        response = client.chat_sync(messages, max_tokens=128)
        print(response[:100] + "..." if len(response) > 100 else response)
        print()


def example_error_handling():
    """
    错误处理示例
    
    演示如何处理常见的错误情况。
    """
    print("=" * 50)
    print("示例 7: 错误处理")
    print("=" * 50)
    
    client = OpenMiniClient("localhost", 50051)
    
    try:
        if not client.health_check():
            print("服务不可用，请检查:")
            print("  1. 服务是否已启动")
            print("  2. 端口是否正确")
            print("  3. 防火墙设置")
            return
        
        messages = [Message(role="user", content="测试消息")]
        
        for response in client.chat(messages, max_tokens=10):
            if "[错误" in response.token:
                print(f"生成错误: {response.token}")
                break
            print(response.token, end="", flush=True)
        print()
        
    except grpc.RpcError as e:
        print(f"gRPC 错误: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"未知错误: {e}")


def run_all_examples():
    """
    运行所有示例
    
    按顺序执行所有功能示例。
    """
    print("\n" + "=" * 50)
    print("OpenMini Python 客户端示例")
    print("=" * 50 + "\n")
    
    example_chat()
    example_sync_chat()
    example_connection_pool()
    example_error_handling()
    
    print("\n提示: 图像理解示例需要测试图片，请确保当前目录有 test_image.jpg")
    print("      或修改示例代码中的图片路径。\n")


if __name__ == "__main__":
    run_all_examples()
