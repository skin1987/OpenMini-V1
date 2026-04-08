# OpenMini API 文档

## 概述

OpenMini 是一个高性能的多模态推理服务，提供 gRPC 接口支持文本对话、图像理解、语音识别等功能。

**服务地址**: `localhost:50051` (默认)

---

## API 端点

### 1. Chat - 文本对话

流式文本对话接口，支持多轮对话和上下文记忆。

**方法**: `rpc Chat(stream ChatRequest) returns (stream ChatResponse)`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| session_id | string | 否 | 会话ID，用于保持对话上下文 |
| messages | Message[] | 是 | 消息列表 |
| stream | bool | 否 | 是否流式输出，默认 true |
| max_tokens | int32 | 否 | 最大生成 token 数，默认 1024 |
| temperature | float | 否 | 温度参数，默认 0.7 |

**Message 结构**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| role | string | 是 | 角色: "user" 或 "assistant" |
| content | string | 是 | 消息内容 |
| image_data | bytes | 否 | 图像数据 (多模态) |
| audio_data | bytes | 否 | 音频数据 |
| video_data | bytes | 否 | 视频数据 |

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话ID |
| token | string | 生成的 token |
| finished | bool | 是否结束 |
| usage | UsageInfo | Token 使用统计 (仅最后一条响应) |

**UsageInfo 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| prompt_tokens | int32 | 输入 token 数 |
| completion_tokens | int32 | 生成 token 数 |
| total_tokens | int32 | 总 token 数 |

#### 示例

```python
from openmini_client import OpenMiniClient, Message

client = OpenMiniClient("localhost", 50051)

messages = [
    Message(role="user", content="你好，请介绍一下自己")
]

for response in client.chat(messages, max_tokens=512, temperature=0.7):
    print(response.token, end="", flush=True)
    if response.finished:
        print(f"\n总 token: {response.usage.total_tokens}")
```

---

### 2. ImageUnderstanding - 图像理解 (非流式)

分析图像内容并回答问题。

**方法**: `rpc ImageUnderstanding(ImageRequest) returns (ImageResponse)`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| session_id | string | 否 | 会话ID |
| image_data | bytes | 是 | 图像二进制数据 |
| question | string | 是 | 关于图像的问题 |
| stream | bool | 否 | 是否流式，默认 false |

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话ID |
| token | string | 完整回答文本 |
| finished | bool | 是否结束 |

#### 示例

```python
with open("image.jpg", "rb") as f:
    image_data = f.read()

answer = client.image_understanding_sync(image_data, "这张图片里有什么？")
print(answer)
```

---

### 3. ImageUnderstandingStream - 图像理解 (流式)

流式返回图像理解结果。

**方法**: `rpc ImageUnderstandingStream(ImageRequest) returns (stream ImageResponse)`

#### 请求参数

同 ImageUnderstanding

#### 响应参数

同 ImageUnderstanding，但为流式响应

#### 示例

```python
with open("photo.jpg", "rb") as f:
    for response in client.image_understanding(f.read(), "描述这张图片"):
        print(response.token, end="", flush=True)
```

---

### 4. HealthCheck - 健康检查

检查服务运行状态。

**方法**: `rpc HealthCheck(HealthRequest) returns (HealthResponse)`

#### 请求参数

空请求

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| healthy | bool | 是否健康 |
| message | string | 状态消息 |

#### 示例

```python
if client.health_check():
    print("服务正常")
else:
    print("服务不可用")
```

---

### 5. OmniChat - 多模态对话

支持文本、图像、音频、视频的多模态对话接口。

**方法**: `rpc OmniChat(stream OmniChatRequest) returns (stream OmniChatResponse)`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| session_id | string | 否 | 会话ID |
| input | oneof | 是 | 输入类型 (见下表) |
| stream | bool | 否 | 是否流式输出 |
| max_tokens | int32 | 否 | 最大生成 token 数 |
| temperature | float | 否 | 温度参数 |

**Input 类型**:

| 字段 | 类型 | 说明 |
|------|------|------|
| text | string | 文本输入 |
| audio_data | bytes | 音频数据 |
| video_data | bytes | 视频数据 |
| image_data | bytes | 图像数据 |

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话ID |
| output | oneof | 输出类型 |
| finished | bool | 是否结束 |
| usage | UsageInfo | Token 统计 |

**Output 类型**:

| 字段 | 类型 | 说明 |
|------|------|------|
| text | string | 文本输出 |
| audio_data | bytes | 音频输出 |

---

### 6. SpeechToText - 语音转文字

将语音转换为文本。

**方法**: `rpc SpeechToText(stream SpeechToTextRequest) returns (stream SpeechToTextResponse)`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| session_id | string | 否 | 会话ID |
| audio_data | bytes | 是 | 音频二进制数据 |
| language | string | 否 | 语言代码，如 "zh", "en" |
| stream | bool | 否 | 是否流式输出 |

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话ID |
| text | string | 转录文本 |
| finished | bool | 是否结束 |
| confidence | float | 置信度 (0-1) |

---

### 7. TextToSpeech - 文字转语音

将文本转换为语音。

**方法**: `rpc TextToSpeech(TextToSpeechRequest) returns (stream TextToSpeechResponse)`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| session_id | string | 否 | 会话ID |
| text | string | 是 | 要转换的文本 |
| voice | string | 否 | 声音类型 |
| language | string | 否 | 语言代码 |
| speed | float | 否 | 语速 (0.5-2.0) |
| pitch | float | 否 | 音调 (0.5-2.0) |

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话ID |
| audio_data | bytes | 音频二进制数据 |
| finished | bool | 是否结束 |

---

## 错误码

| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| OK (0) | 成功 | - |
| CANCELLED (1) | 操作取消 | 重试请求 |
| UNKNOWN (2) | 未知错误 | 检查服务日志 |
| INVALID_ARGUMENT (3) | 无效参数 | 检查请求参数 |
| DEADLINE_EXCEEDED (4) | 请求超时 | 增加 timeout 或减少 max_tokens |
| NOT_FOUND (5) | 资源未找到 | 检查 session_id 或模型路径 |
| ALREADY_EXISTS (6) | 资源已存在 | 使用不同的 session_id |
| PERMISSION_DENIED (7) | 权限不足 | 检查认证信息 |
| RESOURCE_EXHAUSTED (8) | 资源耗尽 | 等待或减少并发 |
| FAILED_PRECONDITION (9) | 前置条件失败 | 检查服务状态 |
| ABORTED (10) | 操作中止 | 重试请求 |
| OUT_OF_RANGE (11) | 超出范围 | 检查参数范围 |
| UNIMPLEMENTED (12) | 功能未实现 | 使用其他接口 |
| INTERNAL (13) | 内部错误 | 检查服务日志 |
| UNAVAILABLE (14) | 服务不可用 | 检查服务是否启动 |
| DATA_LOSS (15) | 数据丢失 | 重试请求 |
| UNAUTHENTICATED (16) | 未认证 | 提供认证信息 |

---

## 常见错误处理

### 1. 连接失败

```
错误: UNAVAILABLE - 服务不可用
```

**解决方案**:
- 检查服务是否启动: `ps aux | grep openmini`
- 检查端口是否监听: `lsof -i :50051`
- 检查防火墙设置

### 2. 内存不足

```
错误: RESOURCE_EXHAUSTED - 内存不足
```

**解决方案**:
- 减小 `max_tokens` 参数
- 降低并发请求数
- 增加系统内存配置

### 3. 请求超时

```
错误: DEADLINE_EXCEEDED - 请求超时
```

**解决方案**:
- 增加 `request_timeout_ms` 配置
- 减少 `max_tokens` 参数
- 检查模型加载状态

### 4. 模型加载失败

```
错误: INTERNAL - 模型加载失败
```

**解决方案**:
- 检查模型文件路径是否正确
- 检查模型文件是否损坏
- 检查内存是否足够加载模型

---

## 性能优化建议

### 1. 连接池配置

```python
# 推荐配置
client = OpenMiniClient(
    host="localhost",
    port=50051,
    pool_size=20  # 根据并发量调整
)
```

### 2. 流式输出

对于长文本生成，推荐使用流式输出:

```python
# 推荐: 流式输出
for response in client.chat(messages):
    process_token(response.token)

# 不推荐: 同步等待完整响应
full_response = client.chat_sync(messages)
```

### 3. 批量请求

对于多个独立请求，使用异步并发:

```python
import asyncio

async def batch_chat(prompts):
    tasks = [client.chat_async([Message("user", p)]) for p in prompts]
    return await asyncio.gather(*tasks)
```

### 4. 内存管理

- 及时清理不需要的会话: `client.clear_memory(session_id)`
- 合理设置 `max_tokens` 避免过度生成
- 使用连接池复用连接

---

## 速率限制

| 端点 | 限制 | 说明 |
|------|------|------|
| Chat | 100 req/s | 每秒请求数 |
| ImageUnderstanding | 20 req/s | 图像处理较慢 |
| SpeechToText | 30 req/s | 音频处理限制 |
| HealthCheck | 1000 req/s | 轻量级请求 |

---

## 版本信息

- **API 版本**: v1.0.0
- **Protocol Buffers**: proto3
- **gRPC 版本**: 0.10+

---

## 联系方式

- **GitHub**: https://github.com/openmini/openmini-v1
- **问题反馈**: https://github.com/openmini/openmini-v1/issues
