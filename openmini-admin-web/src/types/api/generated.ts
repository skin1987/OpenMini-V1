/**
 * 自动生成的 TypeScript 类型定义
 *
 * ⚠️  此文件由 ts-rs 从 Rust 后端自动生成，请勿手动修改！
 *
 * ## 生成来源
 *
 * - openmini-server/src/service/http/types.rs (HTTP REST API)
 * - openmini-server/src/service/grpc/types.rs (gRPC 服务)
 * - openmini-server/src/error.rs (错误类型)
 * - openmini-server/src/config/settings.rs (配置类型)
 * - openmini-server/src/monitoring/health_check.rs (监控类型)
 *
 * ## 如何重新生成
 *
 * ```bash
 * cd openmini-server
 * cargo test --lib types::tests::test_export_types_to_temp_dir
 * ```
 *
 * 或者在程序启动时调用：
 * ```rust
 * use openmini_server::types;
 * types::export_all();
 * ```
 *
 * ## 使用示例
 *
 * ```typescript
 * // 导入 HTTP API 类型
 * import type { ChatCompletionRequest } from './generated/ChatCompletionRequest';
 * import type { ChatCompletionResponse } from './generated/ChatCompletionResponse';
 *
 * // 创建类型安全的请求
 * const request: ChatCompletionRequest = {
 *   session_id: "sess-123",
 *   messages: [
 *     { role: "user", content: "Hello, OpenMini!" }
 *   ],
 *   stream: false,
 *   max_tokens: 1024,
 *   temperature: 0.7,
 * };
 *
 * // 类型安全的响应处理
 * function handleResponse(response: ChatCompletionResponse) {
 *   console.log(`Generated ${response.choices.length} choices`);
 *   response.choices.forEach(choice => {
 *     console.log(choice.message.content);
 *   });
 * }
 * ```
 */

// ========================================================================
// HTTP REST API 类型导出
// ========================================================================

export type { ChatCompletionRequest } from './ChatCompletionRequest';
export type { ChatMessage } from './ChatMessage';
export type { ChatCompletionResponse } from './ChatCompletionResponse';
export type { ChatChoice } from './ChatChoice';
export type { ChatCompletionChunk } from './ChatCompletionChunk';
export type { DeltaChoice } from './DeltaChoice';
export type { DeltaContent } from './DeltaContent';
export type { ImageUnderstandRequest } from './ImageUnderstandRequest';
export type { ImageUnderstandResponse } from './ImageUnderstandResponse';
export type { TtsRequest } from './TtsRequest';
export type { TtsResponse } from './TtsResponse';
export type { SttRequest } from './SttRequest';
export type { SttResponse } from './SttResponse';
export type { UsageInfo } from './UsageInfo';
export type { ModelInfo } from './ModelInfo';
export type { ApiError } from './ApiError';
export type { HealthCheckResponse } from './HealthCheckResponse';

// ========================================================================
// gRPC 服务类型导出
// ========================================================================

export type { GrpcUsageInfo } from './GrpcUsageInfo';
export type { Message } from './Message';
export type { ChatRequest } from './ChatRequest';
export type { ChatResponse } from './ChatResponse';
export type { ImageRequest } from './ImageRequest';
export type { ImageResponse } from './ImageResponse';
export type { HealthRequest } from './HealthRequest';
export type { HealthResponse } from './HealthResponse';
export type { OmniChatRequest } from './OmniChatRequest';
export type { OmniInput } from './OmniInput';
export type { OmniChatResponse } from './OmniChatResponse';
export type { OmniOutput } from './OmniOutput';
export type { SpeechToTextRequest } from './SpeechToTextRequest';
export type { SpeechToTextResponse } from './SpeechToTextResponse';
export type { TextToSpeechRequest } from './TextToSpeechRequest';
export type { TextToSpeechResponse } from './TextToSpeechResponse';

// ========================================================================
// 错误类型导出
// ========================================================================

export type { AppError } from './AppError';
export type { EngineError } from './EngineError';
export type { WorkerError } from './WorkerError';
export type { TrainingError } from './TrainingError';
export type { HardwareError } from './HardwareError';
export type { ConfigError } from './ConfigError';

// ========================================================================
// 配置类型导出
// ========================================================================

export type { ServerConfig } from './ServerConfig';
export type { ServerSettings } from './ServerSettings';
export type { CoreSettings } from './CoreSettings';
export type { ThreadPoolSettings } from './ThreadPoolSettings';
export type { MemorySettings } from './MemorySettings';
export type { ModelSettings } from './ModelSettings';
export type { WorkerSettings } from './WorkerSettings';
export type { GrpcSettings } from './GrpcSettings';
export type { DatabaseSettings } from './DatabaseSettings';

// ========================================================================
// 监控/健康检查类型导出
// ========================================================================

export type { HealthStatus } from './HealthStatus';
export type { ComponentHealth } from './ComponentHealth';
export type { HealthCheckerConfig } from './HealthCheckerConfig';
