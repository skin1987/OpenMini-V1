# OpenMini-V1 Examples (示例集合)

欢迎来到 OpenMini-V1 示例代码库！这里包含了丰富的示例，帮助您快速上手和使用 OpenMini-V1 推理服务。

## 📋 目录

- [快速开始指南](#-5分钟快速开始)
- [示例索引](#-示例索引)
- [常见使用场景](#-常见使用场景)
- [安装与配置](#-安装与配置)
- [故障排除](#-故障排除)

---

## 🚀 5分钟快速开始

### 前置条件

在运行示例之前，请确保：

1. ✅ 已安装 Python 3.10+
2. ✅ OpenMini-V1 服务器正在运行
3. ✅ 已安装依赖包

### 步骤 1: 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install openai httpx aiohttp numpy
```

### 步骤 2: 启动 OpenMini-V1 服务器

```bash
# 从项目根目录启动服务器
cargo run --release -- --model-path ./models/openmini-7b

# 或使用预编译版本
./openmini-server --port 8080
```

### 步骤 3: 运行第一个示例

```bash
# 运行所有示例
cd examples
python python_client.py

# 或运行特定示例（例如示例1：基础对话）
python python_client.py 1
```

### 预期输出

```
╔══════════════════════════════════════════════════════════╗
║         OpenMini-V1 Python Client SDK Examples           ║
╚══════════════════════════════════════════════════════════╝

> Running all examples...

============================================================
[1/9] Basic Chat Completion
============================================================
✅ Server is healthy

📝 Response:
Go语言是一种静态类型、编译型的编程语言...

📊 Token Usage:
   Prompt tokens: 25
   Completion tokens: 128
   Total tokens: 153
   Estimated cost: $0.0002
```

---

## 📚 示例索引

### 示例列表

| 编号 | 示例名称 | 文件 | 描述 | 复杂度 |
|------|---------|------|------|--------|
| **1** | [基础对话完成](#示例1基础对话完成) | `python_client.py` | 简单的问答交互 | ⭐ 初级 |
| **2** | [流式输出](#示例2流式输出) | `python_client.py` | 实时逐字输出 | ⭐⭐ 中级 |
| **3** | [多轮对话](#示例3多轮对话) | `python_client.py` | 维护上下文的连续对话 | ⭐⭐ 中级 |
| **4** | [图像理解](#示例4图像理解视觉) | `python_client.py` | 多模态图像分析 | ⭐⭐⭐ 高级 |
| **5** | [文本向量化](#示例5文本向量化) | `python_client.py` | Embedding 生成与相似度计算 | ⭐⭐ 中级 |
| **6** | [异步客户端](#示例6异步客户端) | `python_client.py` | 并发请求处理 | ⭐⭐⭐ 高级 |
| **7** | [错误处理](#示例7错误处理与重试) | `python_client.py` | 异常捕获与重试机制 | ⭐⭐ 中级 |
| **8** | [性能基准测试](#示例8性能基准测试) | `python_client.py` | 延迟和吞吐量测量 | ⭐⭐⭐ 高级 |
| **9** | [模型列表](#示例9列出可用模型) | `python_client.py` | 查询可用模型信息 | ⭐ 初级 |

### 详细说明

#### 示例 1: 基础对话完成

**功能**: 最简单的问答交互模式

**适用场景**: 单次查询、简单任务、API 测试

```python
from python_client import OpenMiniClient, ChatMessage

with OpenMiniClient(base_url="http://localhost:8080") as client:
    response = client.chat.create(
        model="openmini-7b",
        messages=[
            ChatMessage(role="user", content="你好！")
        ],
    )
    print(response.content)
```

**运行命令**:
```bash
python python_client.py 1
```

---

#### 示例 2: 流式输出

**功能**: 实时显示模型生成的每个 token

**适用场景**: 聊天应用、实时展示、长文本生成

```python
with OpenMiniClient() as client:
    stream = client.chat.create(
        model="openmini-7b",
        messages=[ChatMessage(role="user", content="讲一个故事")],
        stream=True,
    )

    for chunk in stream:
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
```

**运行命令**:
```bash
python python_client.py 2
```

**关键特性**:
- ✅ 逐 token 实时显示
- ✅ 低延迟首字响应
- ✅ 适合聊天界面集成

---

#### 示例 3: 多轮对话

**功能**: 在多次交互中保持上下文连贯性

**适用场景**: 对话系统、客服机器人、教学助手

```python
conversation_history = [
    ChatMessage(role="system", content="你是一个有帮助的助手。"),
]

# 第一轮
conversation_history.append(ChatMessage(role="user", content="什么是AI？"))
response = client.chat.create(messages=conversation_history)
conversation_history.append(ChatMessage(role="assistant", content=response.content))

# 第二轮（自动包含之前的上下文）
conversation_history.append(ChatMessage(role="user", content="能详细解释吗？"))
response = client.chat.create(messages=conversation_history)
```

**运行命令**:
```bash
python python_client.py 3
```

**最佳实践**:
- 💡 设置合理的 `max_tokens` 控制响应长度
- 💡 定期截断历史避免超出上下文窗口
- 💡 使用 system message 定义角色和行为

---

#### 示例 4: 图像理解 (Vision)

**功能**: 分析和理解图片内容

**适用场景**: 图像描述、OCR、视觉问答、内容审核

```python
import base64

# 读取并编码图片
with open("photo.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# 构建多模态消息
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        },
    ]
}]

response = client.chat.create(
    model="openmini-vision-7b",  # 视觉模型
    messages=messages,
)
```

**运行命令**:
```bash
python python_client.py 4

# 或指定图片路径（需要修改代码或扩展参数）
```

**支持的格式**:
- 📷 JPEG, PNG, GIF, WebP
- 📏 Base64 编码或 URL 引用
- 🎯 最大分辨率视模型而定

---

#### 示例 5: 文本向量化

**功能**: 将文本转换为高维向量用于语义搜索

**适用场景**: RAG 系统、推荐引擎、文档聚类、语义相似度

```python
import numpy as np

# 生成向量
response = client.embeddings.create(
    model="openmini-embeddings",
    input=["人工智能", "机器学习", "深度学习"],
)

# 计算相似度
vec1 = np.array(response.embeddings[0])
vec2 = np.array(response.embeddings[1])
similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(f"相似度: {similarity:.4f}")
```

**运行命令**:
```bash
python python_client.py 5
```

**典型应用流程**:
```
用户问题 → 向量化 → 向量数据库检索 → 相关文档 → LLM 回答
```

---

#### 示例 6: 异步客户端

**功能**: 支持高并发请求的异步客户端

**适用场景**: Web 后端服务、批量处理、高吞吐量应用

```python
import asyncio
from python_client import AsyncOpenMiniClient

async def handle_multiple_queries():
    async with AsyncOpenMiniClient() as client:
        # 并发发送多个请求
        tasks = [
            client.chat.create(
                messages=[ChatMessage(role="user", content=q)]
            )
            for q in ["问题1", "问题2", "问题3"]
        ]
        
        results = await asyncio.gather(*tasks)
        for r in results:
            print(r.content)

asyncio.run(handle_multiple_queries())
```

**运行命令**:
```bash
python python_client.py 6
```

**性能优势**:
- ⚡ 并发请求减少总等待时间
- ⚡ 适合 I/O 密集型场景
- ⚡ 与 FastAPI/Starlette 无缝集成

---

#### 示例 7: 错误处理与重试

**功能**: 健壮的错误处理和自动重试机制

**适用场景**: 生产环境、不稳定网络、容错要求高的应用

```python
client = OpenMiniClient(
    base_url="http://localhost:8080",
    max_retries=3,  # 自动重试3次
    timeout=30.0,
)

try:
    response = client.chat.create(...)
except Exception as e:
    print(f"请求失败: {e}")
    # 执行降级逻辑...
```

**运行命令**:
```bash
python python_client.py 7
```

**错误类型**:
| 错误类型 | 描述 | 处理建议 |
|---------|------|---------|
| ConnectionError | 服务器不可达 | 检查网络和服务器状态 |
| TimeoutError | 请求超时 | 增加 timeout 或减少 max_tokens |
| HTTPStatusError | API 错误 | 查看状态码和错误消息 |
| ValidationError | 参数无效 | 检查请求参数格式 |

---

#### 示例 8: 性能基准测试

**功能**: 测量 API 的延迟和吞吐量

**适用场景**: 性能优化、容量规划、SLA 验证

```python
# 自动化基准测试
latencies = []

for i in range(10):
    start = time.time()
    response = client.chat.create(...)
    latency = time.time() - start
    latencies.append(latency)

print(f"平均延迟: {sum(latencies)/len(latencies):.2f}s")
print(f"P99延迟: {sorted(latencies)[int(0.99*len(latencies))]:.2f}s")
```

**运行命令**:
```bash
python python_client.py 8
```

**输出指标**:
- ⏱️ 平均/最小/最大延迟
- 🔢 Token 生成速度 (tokens/s)
- 📊 吞吐量统计

---

#### 示例 9: 列出可用模型

**功能**: 查询服务器上加载的所有模型

**适用场景**: 动态模型选择、运维监控、资源管理

```python
models = client.list_models()

for model in models:
    print(f"模型ID: {model['id']}")
    print(f"所有者: {model.get('owned_by', 'unknown')}")
```

**运行命令**:
```bash
python python_client.py 9
```

**典型输出**:
```
📋 Available Models (3):

   📦 openmini-7b                   | Owner: local
   📦 openmini-vision-7b            | Owner: local
   📦 openmini-embeddings           | Owner: local
```

---

## 🎯 常见使用场景

### 场景 1: 构建聊天机器人

```python
class ChatBot:
    def __init__(self):
        self.client = OpenMiniClient()
        self.history = []
    
    def chat(self, user_input: str) -> str:
        self.history.append(ChatMessage(role="user", content=user_input))
        
        response = self.client.chat.create(
            messages=self.history,
            max_tokens=512,
            temperature=0.7,
        )
        
        self.history.append(
            ChatMessage(role="assistant", content=response.content)
        )
        return response.content
    
    def clear(self):
        """清除对话历史"""
        self.history = []


# 使用
bot = ChatBot()
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    reply = bot.chat(user_input)
    print(f"Bot: {reply}")
```

### 场景 2: 文档问答系统 (RAG)

```python
def rag_qa_system(query: str, documents: list):
    """
    简化的 RAG (Retrieval-Augmented Generation) 流程
    """
    # Step 1: 向量化查询
    query_embedding = client.embeddings.create(input=[query])
    
    # Step 2: 检索相关文档 (简化版 - 实际应使用向量数据库)
    relevant_docs = retrieve_similar_documents(query_embedding, documents)
    
    # Step 3: 构建增强提示
    context = "\n".join(relevant_docs)
    augmented_prompt = f"""基于以下参考文档回答问题。

参考文档：
{context}

问题：{query}
"""
    
    # Step 4: 生成回答
    response = client.chat.create(
        messages=[
            ChatMessage(role="system", content="你是一个文档助手。"),
            ChatMessage(role="user", content=augmented_prompt),
        ],
    )
    
    return response.content


# 使用示例
docs = ["OpenMini-V1 是一个开源推理框架...", "支持多种大语言模型..."]
answer = rag_qa_system("OpenMini-V1 支持哪些模型？", docs)
print(answer)
```

### 场景 3: 批量文本处理

```python
async def batch_process_texts(texts: list[str]) -> list[str]:
    """
    异步批量处理多个文本
    """
    async with AsyncOpenMiniClient() as client:
        tasks = [
            client.chat.create(
                messages=[ChatMessage(
                    role="user", 
                    content=f"总结以下内容：\n{text[:500]}"
                )],
                max_tokens=200,
            )
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        outputs = []
        for text, result in zip(texts, results):
            if isinstance(result, Exception):
                outputs.append(f"[错误] {result}")
            else:
                outputs.append(result.content)
        
        return outputs


# 使用示例
articles = ["文章1的内容...", "文章2的内容...", "文章3的内容..."]
summaries = asyncio.run(batch_process_texts(articles))
for i, summary in enumerate(summaries, 1):
    print(f"\n文章{i}摘要:\n{summary}\n")
```

### 场景 4: 内容审核与分类

```python
def classify_content(text: str) -> dict:
    """
    使用 LLM 进行内容分类和审核
    """
    response = client.chat.create(
        model="openmini-7b",
        messages=[
            ChatMessage(
                role="system",
                content="""你是内容审核助手。请对以下内容进行分类。
返回 JSON 格式：
{
    "category": "技术/娱乐/新闻/其他",
    "sentiment": "正面/中性/负面",
    "is_safe": true/false,
    "confidence": 0.0-1.0
}"""
            ),
            ChatMessage(role="user", content=text),
        ],
        temperature=0.1,  # 低温度以获得确定性输出
        max_tokens=256,
    )
    
    import json
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse classification"}


# 使用示例
result = classify_content("今天天气真好，阳光明媚！")
print(result)
# 输出: {"category": "其他", "sentiment": "正面", "is_safe": true, "confidence": 0.95}
```

---

## ⚙️ 安装与配置

### 环境变量

可以通过环境变量配置客户端行为：

```bash
# ~/.bashrc 或 .env 文件中设置
export OPENMINI_BASE_URL="http://your-server:8080"
export OPENMINI_API_KEY="your-api-key"
export OPENMINI_TIMEOUT=120
export OPENMINI_MAX_RETRIES=3
```

### 配置文件示例 (.env)

```ini
# OpenMini-V1 Configuration
OPENMINI_BASE_URL=http://localhost:8080
OPENMINI_API_KEY=local-dev
OPENMINI_TIMEOUT=120.0
OPENMINI_MAX_RETRIES=3

# Model Settings
DEFAULT_MODEL=openmini-7b
MAX_TOKENS=2048
TEMPERATURE=0.7
```

### 加载配置

```python
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenMiniClient(
    base_url=os.getenv("OPENMINI_BASE_URL", "http://localhost:8080"),
    api_key=os.getenv("OPENMINI_API_KEY", "local-dev"),
)
```

### requirements.txt

创建 `requirements.txt` 文件：

```
openai>=1.0.0
httpx>=0.25.0
aiohttp>=3.9.0
numpy>=1.24.0
python-dotenv>=1.0.0
pytest>=7.4.0
mypy>=1.5.0
ruff>=0.1.0
```

安装：

```bash
pip install -r requirements.txt
```

---

## 🔧 故障排除

### 常见问题及解决方案

#### 问题 1: 连接被拒绝 (Connection Refused)

**症状**:
```
ConnectionRefusedError: [Errno 61] Connection refused
```

**解决方案**:
```bash
# 1. 检查服务器是否运行
curl http://localhost:8080/health

# 2. 如果未运行，启动服务器
cargo run --release

# 3. 检查端口占用
lsof -i :8080
```

---

#### 问题 2: 请求超时 (Timeout)

**症状**:
```
httpx.ReadTimeout: Timed out reading from server
```

**解决方案**:
```python
# 增加超时时间
client = OpenMiniClient(timeout=300.0)  # 5分钟

# 或减少生成的 token 数量
response = client.chat.create(max_tokens=256)  # 减少到256
```

---

#### 问题 3: 内存不足 (OOM)

**症状**:
```
CUDA out of memory
```

**解决方案**:
1. 减小 batch size
2. 使用更小的模型 (如 openmini-7b 替代 openmini-13b)
3. 减少 `max_tokens`
4. 检查 GPU 内存使用: `watch -n 1 nvidia-smi`

---

#### 问题 4: 导入错误 (Import Error)

**症状**:
```
ModuleNotFoundError: No module named 'httpx'
```

**解决方案**:
```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 重新安装依赖
pip install -r requirements.txt

# 检查 Python 版本 (需要 >= 3.10)
python --version
```

---

#### 问题 5: 模型不存在 (Model Not Found)

**症状**:
```
Model 'openmini-7b' not found
```

**解决方案**:
```bash
# 列出可用模型
python python_client.py 9

# 或检查模型路径配置
ls ./models/
```

---

### 调试技巧

启用详细日志输出：

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 或使用 httpx 的日志
import httpx
logging.getLogger("httpx").setLevel(logging.DEBUG)
```

测试连接：

```python
# 快速健康检查
if not client.health_check():
    print("❌ 服务器不可用")
else:
    print("✅ 服务器正常运行")
```

---

## 📖 更多资源

- **完整贡献指南**: [CONTRIBUTING.md](../CONTRIBUTING.md)
- **项目主页**: https://github.com/your-org/OpenMini-V1
- **API 文档**: http://localhost:8080/docs (Swagger UI)
- **问题反馈**: https://github.com/your-org/OpenMini-V1/issues
- **讨论区**: https://github.com/your-org/OpenMini-V1/discussions

---

## 🤝 贡献示例

我们欢迎社区贡献新的示例！如果您有好的使用案例，欢迎提交 PR：

1. Fork 本仓库
2. 在 `examples/` 目录下添加新示例
3. 确保:
   - 代码符合 PEP 8 规范
   - 包含充分的注释和文档字符串
   - 提供运行说明和预期输出
4. 提交 PR 并参考 [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## 📝 更新日志

### v1.0.0 (2026-04-10)

✨ **新功能**:
- 新增完整的 OpenAI 兼容客户端实现
- 支持 9 个实用示例覆盖主要使用场景
- 同步和异步客户端双模式支持
- 完整的类型注解和 IDE 自动补全支持
- 性能基准测试工具
- 错误处理和重试机制

🐛 **修复**:
- 修复流式输出的内存泄漏问题
- 改进错误信息的可读性

---

*最后更新: 2026-04-10*
