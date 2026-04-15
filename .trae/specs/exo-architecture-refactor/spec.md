# EXO架构重构分布式推理系统 Spec

## Why
OpenMini-V1当前的分布式推理实现是简化的、基于配置的静态方案，缺乏自动设备发现、拓扑感知优化、高性能RDMA通信等现代分布式AI推理系统所需的关键能力。EXO项目提供了业界领先的分布式AI推理架构，支持自动设备发现、RDMA over Thunderbolt、拓扑感知自动并行等先进特性。为提升OpenMini-V1的竞争力，需要基于EXO的先进架构对项目进行深度改造。

## What Changes
- **BREAKING**: 重构分布式推理通信层，用EXO的MLX分布式通信替换当前简化的TCP/IP实现
- **BREAKING**: 重构设备管理模块，集成EXO的自动设备发现和拓扑感知能力
- **BREAKING**: 重构并行策略引擎，用EXO的自动并行算法替换当前静态配置方案
- 扩展API兼容性，集成EXO的多协议API支持（OpenAI、Claude、Ollama）
- 重构配置系统，支持EXO的拓扑感知配置和动态优化
- 增强监控和诊断能力，集成EXO的实时性能监控

## Impact
- **Affected specs**: 分布式推理能力、设备管理、通信协议、API兼容性
- **Affected code**: 
  - `src/model/inference/distributed_inference_engine.rs` (完全重构)
  - `src/hardware/gpu/mod.rs` (扩展设备发现)
  - `src/config/settings.rs` (扩展EXO配置)
  - `src/communication/` (新增通信层模块)
  - `src/api/` (扩展多协议支持)
  - `src/monitoring/` (新增性能监控)

## ADDED Requirements

### Requirement: EXO式自动设备发现
系统SHALL自动发现网络中的计算设备并组成集群，无需手动配置设备地址和拓扑。

#### Scenario: 多设备自动发现
- **WHEN** 启动多个运行OpenMini-V1的设备
- **THEN** 设备自动发现彼此并组成统一集群
- **AND** 系统自动建立优化的通信拓扑

#### Scenario: 设备动态加入/退出
- **WHEN** 新设备加入网络
- **THEN** 系统自动发现新设备并重新评估集群拓扑
- **WHEN** 设备异常退出
- **THEN** 系统自动检测并调整任务分配

### Requirement: 拓扑感知自动并行
系统SHALL根据实时设备拓扑和资源状况自动选择最优的并行策略和模型切分方案。

#### Scenario: 异构设备集群
- **WHEN** 集群包含不同性能的GPU设备
- **THEN** 系统根据设备能力自动分配计算负载
- **AND** 选择最优的并行策略（张量并行/流水线并行/混合并行）

#### Scenario: 网络感知优化
- **WHEN** 设备间网络带宽和延迟不同
- **THEN** 系统考虑网络状况优化通信模式和数据传输

### Requirement: 高性能RDMA通信
系统SHALL支持RDMA over Thunderbolt等高性能通信技术，显著降低设备间通信延迟。

#### Scenario: 苹果设备RDMA通信
- **WHEN** 在支持Thunderbolt 5的苹果设备上运行
- **THEN** 系统自动启用RDMA通信
- **AND** 实现微秒级设备间延迟

#### Scenario: 自动降级通信
- **WHEN** 设备不支持RDMA或硬件条件不满足
- **THEN** 系统自动降级到标准TCP/IP通信

### Requirement: 多协议API兼容性
系统SHALL提供与多种AI服务API标准的兼容性，支持现有工具链无缝集成。

#### Scenario: OpenAI API兼容
- **WHEN** 客户端使用OpenAI Chat Completions API格式请求
- **THEN** 系统正确处理请求并返回兼容响应

#### Scenario: Claude API兼容
- **WHEN** 客户端使用Claude Messages API格式请求
- **THEN** 系统正确处理请求并返回兼容响应

#### Scenario: Ollama API兼容
- **WHEN** 客户端使用Ollama API格式请求
- **THEN** 系统正确处理请求并返回兼容响应

## MODIFIED Requirements

### Requirement: 分布式推理配置
现有分布式推理配置系统需要扩展以支持EXO的拓扑感知配置模型。

**修改内容**:
- 从静态配置改为动态拓扑感知配置
- 支持自动并行策略选择，而非手动指定
- 添加设备发现和网络状况配置选项
- 支持RDMA等高级通信协议配置

### Requirement: 设备资源管理
现有设备管理需要重构以支持EXO的自动发现和动态资源评估。

**修改内容**:
- 从手动设备注册改为自动设备发现
- 添加实时资源监控和健康检查
- 支持动态设备加入/退出处理
- 增强故障检测和恢复机制

### Requirement: 通信协议栈
现有通信实现需要完全重构以支持EXO的MLX分布式通信框架。

**修改内容**:
- 用MLX ring和JACCL后端替换当前TCP/IP实现
- 支持RDMA over Thunderbolt高性能通信
- 实现拓扑优化的通信路径选择
- 添加通信性能监控和调优

## REMOVED Requirements

### Requirement: 静态分布式配置
**Reason**: EXO架构采用动态拓扑感知配置，静态配置无法充分利用集群资源和网络状况。
**Migration**: 现有静态配置将转换为动态配置的初始建议值，系统会根据实际状况自动优化。

### Requirement: 手动设备注册
**Reason**: EXO架构支持自动设备发现，手动注册增加运维复杂度且无法处理动态变化。
**Migration**: 现有设备注册配置将转换为设备发现白名单/黑名单配置。

### Requirement: 简化的并行策略
**Reason**: EXO的自动并行算法更加智能，能够根据实时拓扑选择最优策略。
**Migration**: 现有并行策略配置将转换为性能优化建议，实际策略由系统自动选择。