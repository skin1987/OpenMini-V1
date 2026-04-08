"""
OpenMini Chat Interface - OpenMini 对话接口模块

该模块提供了与 OpenMini Rust 推理服务交互的 Python 客户端接口。
通过 gRPC 协议与后端服务通信，支持文本对话和图像理解功能。

主要功能:
    - 文本对话: 支持单轮和多轮对话
    - 图像理解: 支持图像问答
    - 流式输出: 支持流式生成响应

使用示例:
    # 基本文本对话
    chat = OpenMiniChat()
    response = chat.chat([{"role": "user", "content": "你好"}])
    
    # 图像理解
    response = chat.chat_with_image("image.jpg", "描述这张图片")
    
    # 多轮对话
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
        {"role": "user", "content": "介绍一下你自己"}
    ]
    response = chat.multi_turn_chat(messages)

依赖:
    - openmini-client: gRPC 客户端模块
    - 需要运行 openmini-server 服务

作者: OpenMini Team
版本: 0.1.0
"""

import base64
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# 将 openmini-client 目录添加到 Python 路径
# 这样可以直接导入 client 模块
sys.path.insert(0, str(Path(__file__).parent / "openmini-client"))

from client import OpenMiniClient, Message


class OpenMiniChat:
    """
    OpenMini 对话接口类
    
    该类封装了与 OpenMini gRPC 服务的交互逻辑，提供简洁的对话 API。
    支持文本对话、图像理解和多轮对话功能。
    
    Attributes:
        client (OpenMiniClient): gRPC 客户端实例
        session_id (Optional[str]): 当前会话 ID，用于保持对话上下文
    
    Example:
        >>> chat = OpenMiniChat("localhost", 50051)
        >>> response = chat.chat([{"role": "user", "content": "你好"}])
        >>> print(response)
        你好！我是 OpenMini 助手。
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        初始化 OpenMini 对话接口
        
        Args:
            host (str): gRPC 服务地址，默认为 localhost
            port (int): gRPC 服务端口，默认为 50051
        
        Example:
            >>> chat = OpenMiniChat()  # 使用默认地址
            >>> chat = OpenMiniChat("192.168.1.100", 50051)  # 自定义地址
        """
        self.client = OpenMiniClient(host, port)
        self.session_id: Optional[str] = None
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        image_data: Optional[bytes] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        """
        核心对话接口
        
        发送消息到 OpenMini 服务并获取响应。支持流式输出和图像输入。
        
        Args:
            messages (List[Dict[str, Any]]): 消息列表，格式为:
                [{"role": "user/assistant", "content": "消息内容"}]
            image_data (Optional[bytes]): 图像二进制数据，用于图像理解任务
            max_tokens (int): 最大生成 token 数，控制响应长度
            temperature (float): 温度参数，控制生成随机性
                - 0.0: 确定性输出（贪婪采样）
                - 0.7: 平衡创造性和一致性（推荐）
                - 1.0+: 更有创造性但可能不够稳定
            stream (bool): 是否使用流式输出，默认为 True
            
        Returns:
            str: 模型生成的响应文本
            
        Raises:
            grpc.RpcError: gRPC 通信错误
            ValueError: 参数验证错误
            
        Example:
            >>> chat = OpenMiniChat()
            >>> response = chat.chat(
            ...     messages=[{"role": "user", "content": "你好"}],
            ...     max_tokens=512,
            ...     temperature=0.7
            ... )
        """
        # 将消息列表转换为 gRPC 消息格式
        grpc_messages = []
        for msg in messages:
            grpc_messages.append(Message(
                role=msg["role"],
                content=msg["content"],
                # 只在用户消息中附加图像数据
                image_data=image_data if msg["role"] == "user" and image_data else None,
            ))
        
        # 收集流式响应
        result = []
        for response in self.client.chat(
            messages=grpc_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        ):
            result.append(response.token)
            self.session_id = response.session_id
            if response.finished:
                break
        
        return "".join(result)
    
    def chat_with_image(
        self,
        image_path: str,
        question: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        图像对话接口
        
        发送图像和问题到服务，获取图像理解响应。
        
        Args:
            image_path (str): 图像文件路径，支持常见格式（PNG, JPEG, WebP）
            question (str): 关于图像的问题
            max_tokens (int): 最大生成 token 数
            temperature (float): 温度参数
            
        Returns:
            str: 对图像问题的回答
            
        Raises:
            FileNotFoundError: 图像文件不存在
            IOError: 图像文件读取错误
            
        Example:
            >>> chat = OpenMiniChat()
            >>> answer = chat.chat_with_image(
            ...     image_path="photo.jpg",
            ...     question="这张图片里有什么？"
            ... )
        """
        # 读取图像文件为二进制数据
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # 构造消息并调用对话接口
        messages = [{"role": "user", "content": question}]
        return self.chat(messages, image_data, max_tokens, temperature)
    
    def multi_turn_chat(
        self,
        messages: List[Dict[str, Any]],
        image_data: Optional[bytes] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        多轮对话接口
        
        支持包含历史对话记录的多轮对话场景。
        
        Args:
            messages (List[Dict[str, Any]]): 完整对话历史，包含用户和助手的消息
            image_data (Optional[bytes]): 图像二进制数据（可选）
            max_tokens (int): 最大生成 token 数
            temperature (float): 温度参数
            
        Returns:
            str: 基于对话历史的响应
            
        Example:
            >>> chat = OpenMiniChat()
            >>> messages = [
            ...     {"role": "user", "content": "我叫张三"},
            ...     {"role": "assistant", "content": "你好张三！很高兴认识你。"},
            ...     {"role": "user", "content": "我叫什么名字？"}
            ... ]
            >>> response = chat.multi_turn_chat(messages)
            >>> # 响应会包含上下文信息
        """
        return self.chat(messages, image_data, max_tokens, temperature)


def img2base64(file_path: str) -> str:
    """
    将图像文件转换为 base64 编码字符串
    
    用于在需要文本格式传输图像数据时使用。
    
    Args:
        file_path (str): 图像文件路径
        
    Returns:
        str: base64 编码的图像字符串
        
    Example:
        >>> b64_str = img2base64("photo.jpg")
        >>> # 可用于 JSON 传输或 HTML img 标签
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    """
    命令行入口函数
    
    支持两种模式:
    1. 图像模式: 指定 --image 参数，对图像进行问答
    2. 交互模式: 不指定图像，进入交互式对话
    
    命令行参数:
        --host: gRPC 服务地址 (默认: localhost)
        --port: gRPC 服务端口 (默认: 50051)
        --image: 图像文件路径 (可选)
        --question: 问题内容 (默认: "What is interesting about this image?")
        --max-tokens: 最大生成 token 数 (默认: 1024)
        --temperature: 温度参数 (默认: 0.7)
    
    Example:
        # 交互模式
        $ python chat.py
        
        # 图像模式
        $ python chat.py --image photo.jpg --question "描述这张图片"
        
        # 自定义服务地址
        $ python chat.py --host 192.168.1.100 --port 50051
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="OpenMini Chat Interface")
    parser.add_argument("--host", type=str, default="localhost", help="gRPC 服务地址")
    parser.add_argument("--port", type=int, default=50051, help="gRPC 服务端口")
    parser.add_argument("--image", type=str, help="图像文件路径")
    parser.add_argument("--question", type=str, default="What is interesting about this image?", help="问题")
    parser.add_argument("--max-tokens", type=int, default=1024, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    args = parser.parse_args()
    
    # 创建对话实例
    chat_model = OpenMiniChat(args.host, args.port)
    
    if args.image:
        # 图像模式: 对指定图像进行问答
        print(f"Image: {args.image}")
        print(f"Question: {args.question}")
        print("-" * 50)
        
        answer = chat_model.chat_with_image(
            image_path=args.image,
            question=args.question,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"Answer: {answer}")
    else:
        # 交互模式: 进入命令行对话
        print("Interactive Chat Mode (type 'quit' to exit)")
        print("-" * 50)
        
        messages = []
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() == "quit":
                break
            
            # 添加用户消息
            messages.append({"role": "user", "content": user_input})
            
            # 获取响应
            answer = chat_model.chat(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            
            # 显示并保存助手响应
            print(f"Assistant: {answer}")
            messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
