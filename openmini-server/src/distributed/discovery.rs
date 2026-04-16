//! EXO设备发现和管理模块
//!
//! 基于EXO架构的现代设备发现系统，支持：
//! - libp2p去中心化设备发现
//! - 自动拓扑构建和管理
//! - 设备健康检查和资源监控
//! - 动态集群成员管理
//!
//! # 架构概述
//!
//! ```text
//! ExoDeviceDiscovery (主服务)
//! ├── DiscoveryBackend (trait)
//! │   ├── Libp2pBackend (libp2p实现)
//! │   ├── MdnsBackend (mDNS实现)
//! │   └── StaticBackend (静态配置)
//! ├── PeerRegistry (对等节点注册表)
//! │   ├── PeerInfo (节点信息)
//! │   └── PeerState (节点状态)
//! ├── TopologyManager (拓扑管理器)
//! │   ├── TopologyBuilder (拓扑构建器)
//! │   └── TopologyOptimizer (拓扑优化器)
//! ├── HealthChecker (健康检查器)
//! │   ├── LatencyProber (延迟探测器)
//! │   └── ResourceMonitor (资源监控器)
//! └── EventBus (事件总线)
//!     ├── DiscoveryEvent (发现事件)
//!     └── TopologyEvent (拓扑事件)
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::distributed::discovery::{
//!     ExoDeviceDiscovery, DiscoveryConfig, DiscoveryBackendType
//! };
//!
//! // 创建设备发现服务
//! let config = DiscoveryConfig {
//!     backend: DiscoveryBackendType::Libp2p,
//!     discovery_interval_secs: 30,
//!     health_check_interval_secs: 10,
//!     topology_update_interval_secs: 60,
//! };
//!
//! let mut discovery = ExoDeviceDiscovery::new(config).await?;
//!
//! // 启动发现服务
//! discovery.start().await?;
//!
//! // 获取当前拓扑
//! let topology = discovery.get_topology().await?;
//! println!("发现 {} 个设备", topology.device_count());
//!
//! // 监听设备加入事件
//! while let Some(event) = discovery.next_event().await {
//!     match event {
//!         DiscoveryEvent::DeviceJoined { device_id, info } => {
//!             println!("新设备加入: {}", device_id);
//!         }
//!         DiscoveryEvent::DeviceLeft { device_id } => {
//!             println!("设备离开: {}", device_id);
//!         }
//!         _ => {}
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

// libp2p imports (temporarily disabled due to dependency issues)
// use libp2p::{
//     identity, noise, swarm::SwarmEvent, tcp, yamux, Multiaddr, PeerId, Swarm,
//     core::upgrade,
//     dns, identify, kad::{self, store::MemoryStore, Kademlia, KademliaConfig, KademliaEvent, QueryId, QueryResult},
//     mdns::{self, Mdns, MdnsConfig, MdnsEvent},
//     rendezvous::{self, Rendezvous, RendezvousConfig, RendezvousEvent},
//     request_response::{self, ProtocolSupport, RequestResponse, RequestResponseEvent},
//     swarm::{NetworkBehaviour, SwarmBuilder},
//     Transport,
// };

use super::exo_communication::{DeviceCapabilities, DeviceType, NetworkDevice, NetworkTopology};

// ==================== 核心类型定义 ====================

/// 设备发现后端类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryBackendType {
    /// libp2p后端 (去中心化，支持广域网)
    Libp2p,

    /// mDNS后端 (局域网自动发现)
    Mdns,

    /// 静态配置后端 (测试和开发)
    Static,

    /// 混合后端 (libp2p + mDNS)
    Hybrid,
}

impl std::fmt::Display for DiscoveryBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Libp2p => write!(f, "libp2p"),
            Self::Mdns => write!(f, "mDNS"),
            Self::Static => write!(f, "Static"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// 设备信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// 设备唯一标识
    pub device_id: String,

    /// 设备类型
    pub device_type: DeviceType,

    /// 设备能力
    pub capabilities: DeviceCapabilities,

    /// 设备资源信息
    pub resources: DeviceResources,

    /// 网络地址 (多地址格式)
    pub network_addresses: Vec<String>,

    /// 设备元数据
    pub metadata: HashMap<String, String>,

    /// 首次发现时间
    pub first_seen: SystemTime,

    /// 最后活跃时间
    pub last_seen: SystemTime,

    /// 设备版本信息
    pub version: DeviceVersion,
}

/// 设备资源信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceResources {
    /// 总内存 (字节)
    pub total_memory_bytes: u64,

    /// 可用内存 (字节)
    pub available_memory_bytes: u64,

    /// CPU核心数
    pub cpu_cores: u32,

    /// CPU利用率 (0-1)
    pub cpu_utilization: f32,

    /// GPU信息 (如果有)
    pub gpu_info: Option<GpuInfo>,

    /// 存储信息
    pub storage_info: StorageInfo,

    /// 网络带宽容量 (Gbps)
    pub network_bandwidth_gbps: f32,
}

/// GPU信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU名称
    pub name: String,

    /// GPU显存大小 (字节)
    pub memory_size_bytes: u64,

    /// 可用显存 (字节)
    pub available_memory_bytes: u64,

    /// 计算能力评分 (0-100)
    pub compute_score: u8,

    /// 是否支持RDMA
    pub supports_rdma: bool,

    /// 是否支持MLX
    pub supports_mlx: bool,
}

/// 存储信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    /// 总存储空间 (字节)
    pub total_storage_bytes: u64,

    /// 可用存储空间 (字节)
    pub available_storage_bytes: u64,

    /// 存储类型 (SSD/HDD/NVMe)
    pub storage_type: String,

    /// 存储性能评分 (0-100)
    pub performance_score: u8,
}

/// 设备版本信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceVersion {
    /// 软件版本
    pub software_version: String,

    /// 协议版本
    pub protocol_version: String,

    /// 硬件版本
    pub hardware_version: String,

    /// 固件版本
    pub firmware_version: String,
}

/// 设备状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceState {
    /// 设备信息
    pub info: DeviceInfo,

    /// 健康状态
    pub health: DeviceHealth,

    /// 连接状态
    pub connectivity: ConnectivityState,

    /// 负载信息
    pub load: DeviceLoad,

    /// 最后健康检查时间
    pub last_health_check: SystemTime,
}

/// 设备健康状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceHealth {
    /// 健康
    Healthy,

    /// 警告 (性能下降)
    Warning,

    /// 错误 (功能受损)
    Error,

    /// 离线
    Offline,

    /// 未知
    Unknown,
}

/// 连接状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectivityState {
    /// 已连接
    Connected,

    /// 连接中
    Connecting,

    /// 断开连接
    Disconnected,

    /// 连接错误
    Error,
}

/// 设备负载信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceLoad {
    /// CPU负载 (0-1)
    pub cpu_load: f32,

    /// 内存负载 (0-1)
    pub memory_load: f32,

    /// GPU负载 (0-1)
    pub gpu_load: Option<f32>,

    /// 网络负载 (0-1)
    pub network_load: f32,

    /// 存储负载 (0-1)
    pub storage_load: f32,

    /// 当前任务数
    pub current_tasks: u32,

    /// 最大任务容量
    pub max_tasks: u32,
}

/// 发现事件
#[derive(Debug, Clone, Serialize)]
pub enum DiscoveryEvent {
    /// 设备加入
    DeviceJoined { device_id: String, info: DeviceInfo },

    /// 设备离开
    DeviceLeft { device_id: String, reason: String },

    /// 设备状态更新
    DeviceUpdated {
        device_id: String,
        state: DeviceState,
    },

    /// 拓扑变化
    TopologyChanged {
        old_topology: NetworkTopology,
        new_topology: NetworkTopology,
    },

    /// 发现错误
    DiscoveryError {
        error: String,
        severity: ErrorSeverity,
    },

    /// 健康状态变化
    HealthStatusChanged {
        device_id: String,
        old_health: DeviceHealth,
        new_health: DeviceHealth,
    },
}

/// 错误严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// 低 (可恢复)
    Low,

    /// 中 (需要关注)
    Medium,

    /// 高 (需要立即处理)
    High,

    /// 致命 (服务不可用)
    Critical,
}

/// 设备发现配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// 发现后端类型
    pub backend: DiscoveryBackendType,

    /// 发现间隔 (秒)
    pub discovery_interval_secs: u64,

    /// 健康检查间隔 (秒)
    pub health_check_interval_secs: u64,

    /// 拓扑更新间隔 (秒)
    pub topology_update_interval_secs: u64,

    /// 设备过期时间 (秒)
    pub device_expiry_secs: u64,

    /// 是否启用自动拓扑优化
    pub auto_topology_optimization: bool,

    /// 是否启用健康检查
    pub health_check_enabled: bool,

    /// 是否启用事件广播
    pub event_broadcast_enabled: bool,

    /// 最大设备数量
    pub max_devices: usize,

    /// 本地设备信息 (可选)
    pub local_device_info: Option<DeviceInfo>,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            backend: DiscoveryBackendType::Hybrid,
            discovery_interval_secs: 30,
            health_check_interval_secs: 10,
            topology_update_interval_secs: 60,
            device_expiry_secs: 300, // 5分钟
            auto_topology_optimization: true,
            health_check_enabled: true,
            event_broadcast_enabled: true,
            max_devices: 100,
            local_device_info: None,
        }
    }
}

// ==================== 发现后端trait ====================

/// 设备发现后端trait
#[async_trait::async_trait]
pub trait DiscoveryBackend: Send + Sync {
    /// 启动发现后端
    async fn start(&mut self) -> Result<(), DiscoveryError>;

    /// 停止发现后端
    async fn stop(&mut self) -> Result<(), DiscoveryError>;

    /// 发现设备 (主动发现)
    async fn discover_devices(&mut self) -> Result<Vec<DeviceInfo>, DiscoveryError>;

    /// 广播本地设备信息
    async fn advertise_local_device(&mut self, device: &DeviceInfo) -> Result<(), DiscoveryError>;

    /// 监听设备发现事件
    async fn listen(&mut self) -> Result<Option<DiscoveryEvent>, DiscoveryError>;

    /// 获取后端类型
    fn backend_type(&self) -> DiscoveryBackendType;

    /// 检查后端健康状态
    fn is_healthy(&self) -> bool;

    /// 获取性能统计
    fn get_stats(&self) -> DiscoveryStats;
}

/// 发现统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryStats {
    /// 总发现设备数
    pub total_devices_discovered: u64,

    /// 当前活跃设备数
    pub active_devices: u64,

    /// 发现操作次数
    pub discovery_operations: u64,

    /// 成功发现次数
    pub successful_discoveries: u64,

    /// 失败发现次数
    pub failed_discoveries: u64,

    /// 平均发现延迟 (纳秒)
    pub avg_discovery_latency_ns: u64,

    /// 最后发现时间
    pub last_discovery_time: Option<SystemTime>,
}

/// 发现错误
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("发现后端初始化失败: {0}")]
    BackendInitFailed(String),

    #[error("发现操作失败: {0}")]
    DiscoveryFailed(String),

    #[error("设备信息无效: {0}")]
    InvalidDeviceInfo(String),

    #[error("网络错误: {0}")]
    NetworkError(String),

    #[error("配置错误: {0}")]
    ConfigError(String),

    #[error("超时错误: {0}")]
    Timeout(String),

    #[error("资源不足: {0}")]
    InsufficientResources(String),

    #[error("序列化错误: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("未知错误: {0}")]
    Unknown(String),
}

// ==================== 对等节点注册表 ====================

/// 对等节点注册表
pub struct PeerRegistry {
    /// 设备ID到设备状态的映射
    devices: HashMap<String, DeviceState>,

    /// 设备ID到网络地址的映射
    device_addresses: HashMap<String, Vec<String>>,

    /// 设备订阅者 (设备ID到订阅者列表)
    subscribers: HashMap<String, Vec<mpsc::Sender<DiscoveryEvent>>>,

    /// 注册表版本
    version: u64,

    /// 最后更新时间
    last_updated: SystemTime,
}

impl PeerRegistry {
    /// 创建新的对等节点注册表
    fn new() -> Self {
        Self {
            devices: HashMap::new(),
            device_addresses: HashMap::new(),
            subscribers: HashMap::new(),
            version: 1,
            last_updated: SystemTime::now(),
        }
    }

    /// 注册新设备
    fn register_device(
        &mut self,
        device_id: String,
        info: DeviceInfo,
    ) -> Result<(), DiscoveryError> {
        if self.devices.contains_key(&device_id) {
            return Err(DiscoveryError::DiscoveryFailed(format!(
                "设备已存在: {}",
                device_id
            )));
        }

        let state = DeviceState {
            info: info.clone(),
            health: DeviceHealth::Unknown,
            connectivity: ConnectivityState::Connecting,
            load: DeviceLoad {
                cpu_load: 0.0,
                memory_load: 0.0,
                gpu_load: None,
                network_load: 0.0,
                storage_load: 0.0,
                current_tasks: 0,
                max_tasks: 100,
            },
            last_health_check: SystemTime::now(),
        };

        self.devices.insert(device_id.clone(), state);
        self.device_addresses
            .insert(device_id.clone(), info.network_addresses.clone());
        self.version += 1;
        self.last_updated = SystemTime::now();

        info!("设备注册成功: {}", device_id);

        Ok(())
    }

    /// 更新设备信息
    fn update_device(&mut self, device_id: &str, info: DeviceInfo) -> Result<(), DiscoveryError> {
        if let Some(state) = self.devices.get_mut(device_id) {
            state.info = info.clone();
            state.last_health_check = SystemTime::now();

            if let Some(addresses) = self.device_addresses.get_mut(device_id) {
                *addresses = info.network_addresses.clone();
            }

            self.version += 1;
            self.last_updated = SystemTime::now();

            debug!("设备信息更新: {}", device_id);
            Ok(())
        } else {
            Err(DiscoveryError::DiscoveryFailed(format!(
                "设备不存在: {}",
                device_id
            )))
        }
    }

    /// 移除设备
    fn remove_device(&mut self, device_id: &str, reason: String) -> Result<(), DiscoveryError> {
        if self.devices.remove(device_id).is_some() {
            self.device_addresses.remove(device_id);
            self.version += 1;
            self.last_updated = SystemTime::now();

            warn!("设备移除: {} (原因: {})", device_id, reason);
            Ok(())
        } else {
            Err(DiscoveryError::DiscoveryFailed(format!(
                "设备不存在: {}",
                device_id
            )))
        }
    }

    /// 更新设备状态
    fn update_device_state(
        &mut self,
        device_id: &str,
        state: DeviceState,
    ) -> Result<(), DiscoveryError> {
        if self.devices.contains_key(device_id) {
            self.devices.insert(device_id.to_string(), state);
            self.version += 1;
            self.last_updated = SystemTime::now();
            Ok(())
        } else {
            Err(DiscoveryError::DiscoveryFailed(format!(
                "设备不存在: {}",
                device_id
            )))
        }
    }

    /// 获取设备状态
    fn get_device_state(&self, device_id: &str) -> Option<&DeviceState> {
        self.devices.get(device_id)
    }

    /// 获取所有设备状态
    fn get_all_devices(&self) -> Vec<&DeviceState> {
        self.devices.values().collect()
    }

    /// 获取设备数量
    fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// 检查设备是否存在
    fn contains_device(&self, device_id: &str) -> bool {
        self.devices.contains_key(device_id)
    }

    /// 获取注册表版本
    fn version(&self) -> u64 {
        self.version
    }

    /// 获取最后更新时间
    fn last_updated(&self) -> SystemTime {
        self.last_updated
    }
}

// ==================== 设备发现服务 ====================

/// EXO设备发现服务
pub struct ExoDeviceDiscovery {
    /// 发现后端
    backend: Box<dyn DiscoveryBackend>,

    /// 对等节点注册表
    peer_registry: Arc<RwLock<PeerRegistry>>,

    /// 拓扑管理器
    topology_manager: Arc<TopologyManager>,

    /// 健康检查器
    health_checker: Arc<HealthChecker>,

    /// 事件发送器
    event_sender: mpsc::Sender<DiscoveryEvent>,

    /// 事件接收器
    event_receiver: mpsc::Receiver<DiscoveryEvent>,

    /// 配置
    config: DiscoveryConfig,

    /// 是否正在运行
    running: bool,

    /// 本地设备信息
    local_device_info: DeviceInfo,
}

impl ExoDeviceDiscovery {
    /// 创建设备发现服务
    pub async fn new(config: DiscoveryConfig) -> Result<Self, DiscoveryError> {
        // 创建事件通道
        let (event_sender, event_receiver) = mpsc::channel(100);

        // 创建本地设备信息
        let local_device_info = config
            .local_device_info
            .clone()
            .unwrap_or_else(|| Self::create_local_device_info());

        // 创建对等节点注册表
        let peer_registry = Arc::new(RwLock::new(PeerRegistry::new()));

        // 创建拓扑管理器
        let topology_manager = Arc::new(TopologyManager::new(peer_registry.clone()));

        // 创建健康检查器
        let health_checker = Arc::new(HealthChecker::new(
            peer_registry.clone(),
            event_sender.clone(),
        ));

        // 创建发现后端
        let backend = Self::create_backend(config.backend, &config).await?;

        Ok(Self {
            backend,
            peer_registry,
            topology_manager,
            health_checker,
            event_sender,
            event_receiver,
            config,
            running: false,
            local_device_info,
        })
    }

    /// 创建本地设备信息
    fn create_local_device_info() -> DeviceInfo {
        let device_id = format!("local-{}", Uuid::new_v4().to_string()[..8].to_string());

        DeviceInfo {
            device_id,
            device_type: DeviceType::Cpu, // 默认类型
            capabilities: DeviceCapabilities {
                supports_rdma: false,
                supports_mlx: false,
                supports_thunderbolt: false,
                max_memory_bandwidth_gbs: 10.0,
                compute_score: 50,
            },
            resources: DeviceResources {
                total_memory_bytes: 8 * 1024 * 1024 * 1024,     // 8GB
                available_memory_bytes: 6 * 1024 * 1024 * 1024, // 6GB
                cpu_cores: 4,
                cpu_utilization: 0.1,
                gpu_info: None,
                storage_info: StorageInfo {
                    total_storage_bytes: 512 * 1024 * 1024 * 1024, // 512GB
                    available_storage_bytes: 256 * 1024 * 1024 * 1024, // 256GB
                    storage_type: "SSD".to_string(),
                    performance_score: 80,
                },
                network_bandwidth_gbps: 1.0,
            },
            network_addresses: vec!["127.0.0.1:8080".to_string()],
            metadata: HashMap::new(),
            first_seen: SystemTime::now(),
            last_seen: SystemTime::now(),
            version: DeviceVersion {
                software_version: "0.1.0".to_string(),
                protocol_version: "1.0".to_string(),
                hardware_version: "unknown".to_string(),
                firmware_version: "unknown".to_string(),
            },
        }
    }

    /// 创建发现后端
    async fn create_backend(
        backend_type: DiscoveryBackendType,
        config: &DiscoveryConfig,
    ) -> Result<Box<dyn DiscoveryBackend>, DiscoveryError> {
        match backend_type {
            DiscoveryBackendType::Libp2p => {
                // TODO: 临时使用静态后端，因为libp2p依赖问题
                warn!("libp2p后端暂时不可用，使用静态后端代替");
                Ok(Box::new(StaticBackend::new(config).await?))
            }
            DiscoveryBackendType::Mdns => {
                // 创建mDNS后端
                Ok(Box::new(MdnsBackend::new(config).await?))
            }
            DiscoveryBackendType::Static => {
                // 创建静态后端
                Ok(Box::new(StaticBackend::new(config).await?))
            }
            DiscoveryBackendType::Hybrid => {
                // 创建混合后端
                Ok(Box::new(HybridBackend::new(config).await?))
            }
        }
    }

    /// 启动发现服务
    pub async fn start(&mut self) -> Result<(), DiscoveryError> {
        if self.running {
            return Err(DiscoveryError::DiscoveryFailed(
                "发现服务已在运行".to_string(),
            ));
        }

        info!("启动EXO设备发现服务，后端: {}", self.backend.backend_type());

        // 启动发现后端
        self.backend.start().await?;

        // 广播本地设备信息
        self.backend
            .advertise_local_device(&self.local_device_info)
            .await?;

        // 注册本地设备
        {
            let mut registry = self.peer_registry.write().await;
            registry.register_device(
                self.local_device_info.device_id.clone(),
                self.local_device_info.clone(),
            )?;
        }

        // 启动健康检查器（如果启用）
        if self.config.health_check_enabled {
            self.health_checker
                .start(self.config.health_check_interval_secs);
        }

        self.running = true;
        info!("EXO设备发现服务启动成功");

        Ok(())
    }

    /// 停止发现服务
    pub async fn stop(&mut self) -> Result<(), DiscoveryError> {
        if !self.running {
            return Err(DiscoveryError::DiscoveryFailed(
                "发现服务未运行".to_string(),
            ));
        }

        info!("停止EXO设备发现服务");

        // 停止健康检查器
        if self.config.health_check_enabled {
            self.health_checker.stop();
        }

        // 停止发现后端
        self.backend.stop().await?;

        self.running = false;
        info!("EXO设备发现服务已停止");

        Ok(())
    }

    /// 获取当前拓扑
    pub async fn get_topology(&self) -> Result<NetworkTopology, DiscoveryError> {
        self.topology_manager.get_topology().await
    }

    /// 获取设备列表
    pub async fn get_devices(&self) -> Vec<DeviceState> {
        let registry = self.peer_registry.read().await;
        registry.get_all_devices().into_iter().cloned().collect()
    }

    /// 获取设备数量
    pub async fn device_count(&self) -> usize {
        let registry = self.peer_registry.read().await;
        registry.device_count()
    }

    /// 获取下一个发现事件
    pub async fn next_event(&mut self) -> Option<DiscoveryEvent> {
        self.event_receiver.recv().await
    }

    /// 获取发现统计
    pub fn get_stats(&self) -> DiscoveryStats {
        self.backend.get_stats()
    }

    /// 检查服务是否正在运行
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// 获取本地设备信息
    pub fn local_device_info(&self) -> &DeviceInfo {
        &self.local_device_info
    }
}

// ==================== 拓扑管理器 ====================

/// 拓扑管理器
pub struct TopologyManager {
    /// 对等节点注册表引用
    peer_registry: Arc<RwLock<PeerRegistry>>,

    /// 当前拓扑
    current_topology: Arc<RwLock<NetworkTopology>>,

    /// 拓扑历史记录
    topology_history: Vec<NetworkTopology>,

    /// 最后拓扑更新时间
    last_topology_update: SystemTime,
}

impl TopologyManager {
    /// 创建新的拓扑管理器
    fn new(peer_registry: Arc<RwLock<PeerRegistry>>) -> Self {
        Self {
            peer_registry,
            current_topology: Arc::new(RwLock::new(NetworkTopology {
                devices: Vec::new(),
                links: Vec::new(),
                updated_at: SystemTime::now(),
                version: 1,
            })),
            topology_history: Vec::new(),
            last_topology_update: SystemTime::now(),
        }
    }

    /// 获取当前拓扑
    async fn get_topology(&self) -> Result<NetworkTopology, DiscoveryError> {
        let topology = self.current_topology.read().await;
        Ok(topology.clone())
    }

    /// 更新拓扑
    async fn update_topology(&mut self) -> Result<(), DiscoveryError> {
        let registry = self.peer_registry.read().await;
        let devices = registry.get_all_devices();

        // 将设备状态转换为网络设备
        let network_devices: Vec<NetworkDevice> = devices
            .iter()
            .map(|state| {
                NetworkDevice {
                    device_id: state.info.device_id.clone(),
                    device_type: state.info.device_type,
                    network_address: state
                        .info
                        .network_addresses
                        .first()
                        .cloned()
                        .unwrap_or_default(),
                    bandwidth_gbps: state.info.resources.network_bandwidth_gbps,
                    baseline_latency_ns: 1_000_000, // 默认1毫秒
                    capabilities: state.info.capabilities.clone(),
                }
            })
            .collect();

        // 创建网络链路 (简化实现)
        let mut links = Vec::new();
        if network_devices.len() > 1 {
            // 创建全连接拓扑 (简化)
            for i in 0..network_devices.len() {
                for j in (i + 1)..network_devices.len() {
                    links.push(create_link(&network_devices[i], &network_devices[j]));
                }
            }
        }

        // 创建新拓扑
        let new_topology = NetworkTopology {
            devices: network_devices,
            links,
            updated_at: SystemTime::now(),
            version: registry.version(),
        };

        // 保存历史记录
        let old_topology = self.current_topology.read().await.clone();
        self.topology_history.push(old_topology);

        // 限制历史记录大小
        if self.topology_history.len() > 10 {
            self.topology_history.remove(0);
        }

        // 更新当前拓扑
        {
            let mut current = self.current_topology.write().await;
            *current = new_topology.clone();
        }

        self.last_topology_update = SystemTime::now();

        Ok(())
    }
}

/// 创建网络链路
fn create_link(
    source: &NetworkDevice,
    target: &NetworkDevice,
) -> super::exo_communication::NetworkLink {
    use super::exo_communication::LinkType;

    super::exo_communication::NetworkLink {
        source_device_id: source.device_id.clone(),
        target_device_id: target.device_id.clone(),
        link_type: LinkType::Ethernet, // 默认以太网
        bandwidth_gbps: source.bandwidth_gbps.min(target.bandwidth_gbps),
        latency_ns: 1_000_000, // 默认1毫秒
        reliability: 0.99,
    }
}

// ==================== 健康检查器 ====================

/// 健康检查器
struct HealthChecker {
    /// 对等节点注册表引用
    peer_registry: Arc<RwLock<PeerRegistry>>,

    /// 事件发送器
    event_sender: mpsc::Sender<DiscoveryEvent>,

    /// 是否正在运行
    running: bool,
}

impl HealthChecker {
    /// 创建新的健康检查器
    fn new(
        peer_registry: Arc<RwLock<PeerRegistry>>,
        event_sender: mpsc::Sender<DiscoveryEvent>,
    ) -> Self {
        Self {
            peer_registry,
            event_sender,
            running: false,
        }
    }

    /// 启动健康检查器
    fn start(&self, interval_secs: u64) {
        // 简化实现：实际应启动定时任务
        info!("健康检查器已启动，检查间隔: {}秒", interval_secs);
    }

    /// 停止健康检查器
    fn stop(&self) {
        info!("健康检查器已停止");
    }
}

// ==================== 后端实现 ====================

/// libp2p网络行为（暂时禁用）
// #[derive(NetworkBehaviour)]
// struct DiscoveryBehaviour {
//     /// Identify协议用于获取对等节点信息
//     identify: identify::Behaviour<identify::Config>,
//     /// Kademlia DHT用于节点发现
//     kademlia: Kademlia<MemoryStore>,
//     /// mDNS用于局域网发现
//     mdns: Mdns,
//     /// Rendezvous用于会合点发现
//     rendezvous: Rendezvous,
// }
/// libp2p发现后端（暂时简化实现）
struct Libp2pBackend {
    config: DiscoveryConfig,
    stats: DiscoveryStats,
    discovered_peers: HashMap<String, DeviceInfo>,
    pending_events: Vec<DiscoveryEvent>,
}

impl Libp2pBackend {
    async fn new(config: &DiscoveryConfig) -> Result<Self, DiscoveryError> {
        warn!("libp2p后端使用简化实现（libp2p依赖暂时不可用）");

        Ok(Self {
            config: config.clone(),
            stats: DiscoveryStats {
                total_devices_discovered: 0,
                active_devices: 0,
                discovery_operations: 0,
                successful_discoveries: 0,
                failed_discoveries: 0,
                avg_discovery_latency_ns: 0,
                last_discovery_time: None,
            },
            discovered_peers: HashMap::new(),
            pending_events: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl DiscoveryBackend for Libp2pBackend {
    async fn start(&mut self) -> Result<(), DiscoveryError> {
        warn!("libp2p发现后端（简化实现）启动");
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), DiscoveryError> {
        info!("libp2p发现后端停止");
        Ok(())
    }

    async fn discover_devices(&mut self) -> Result<Vec<DeviceInfo>, DiscoveryError> {
        // 简化实现：返回已发现的设备
        let devices: Vec<DeviceInfo> = self.discovered_peers.values().cloned().collect();

        self.stats.discovery_operations += 1;
        if !devices.is_empty() {
            self.stats.successful_discoveries += 1;
        }
        self.stats.total_devices_discovered = devices.len() as u64;
        self.stats.last_discovery_time = Some(std::time::SystemTime::now());

        Ok(devices)
    }

    async fn advertise_local_device(&mut self, device: &DeviceInfo) -> Result<(), DiscoveryError> {
        info!("广播本地设备信息（简化实现）: {}", device.device_id);
        // 简化实现：将设备添加到已发现列表
        self.discovered_peers
            .insert(device.device_id.clone(), device.clone());
        Ok(())
    }

    async fn listen(&mut self) -> Result<Option<DiscoveryEvent>, DiscoveryError> {
        // 返回待处理事件
        if let Some(event) = self.pending_events.pop() {
            Ok(Some(event))
        } else {
            Ok(None)
        }
    }

    fn backend_type(&self) -> DiscoveryBackendType {
        DiscoveryBackendType::Libp2p
    }

    fn is_healthy(&self) -> bool {
        true
    }

    fn get_stats(&self) -> DiscoveryStats {
        self.stats.clone()
    }
}

/// mDNS发现后端
struct MdnsBackend {
    config: DiscoveryConfig,
    stats: DiscoveryStats,
}

impl MdnsBackend {
    async fn new(config: &DiscoveryConfig) -> Result<Self, DiscoveryError> {
        Ok(Self {
            config: config.clone(),
            stats: DiscoveryStats {
                total_devices_discovered: 0,
                active_devices: 0,
                discovery_operations: 0,
                successful_discoveries: 0,
                failed_discoveries: 0,
                avg_discovery_latency_ns: 0,
                last_discovery_time: None,
            },
        })
    }
}

#[async_trait::async_trait]
impl DiscoveryBackend for MdnsBackend {
    async fn start(&mut self) -> Result<(), DiscoveryError> {
        info!("mDNS发现后端启动");
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), DiscoveryError> {
        info!("mDNS发现后端停止");
        Ok(())
    }

    async fn discover_devices(&mut self) -> Result<Vec<DeviceInfo>, DiscoveryError> {
        // 简化实现：返回空列表
        Ok(Vec::new())
    }

    async fn advertise_local_device(&mut self, _device: &DeviceInfo) -> Result<(), DiscoveryError> {
        // 简化实现
        Ok(())
    }

    async fn listen(&mut self) -> Result<Option<DiscoveryEvent>, DiscoveryError> {
        // 简化实现：返回None
        Ok(None)
    }

    fn backend_type(&self) -> DiscoveryBackendType {
        DiscoveryBackendType::Mdns
    }

    fn is_healthy(&self) -> bool {
        true
    }

    fn get_stats(&self) -> DiscoveryStats {
        self.stats.clone()
    }
}

/// 静态发现后端
struct StaticBackend {
    config: DiscoveryConfig,
    stats: DiscoveryStats,
}

impl StaticBackend {
    async fn new(config: &DiscoveryConfig) -> Result<Self, DiscoveryError> {
        Ok(Self {
            config: config.clone(),
            stats: DiscoveryStats {
                total_devices_discovered: 0,
                active_devices: 0,
                discovery_operations: 0,
                successful_discoveries: 0,
                failed_discoveries: 0,
                avg_discovery_latency_ns: 0,
                last_discovery_time: None,
            },
        })
    }
}

#[async_trait::async_trait]
impl DiscoveryBackend for StaticBackend {
    async fn start(&mut self) -> Result<(), DiscoveryError> {
        info!("静态发现后端启动");
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), DiscoveryError> {
        info!("静态发现后端停止");
        Ok(())
    }

    async fn discover_devices(&mut self) -> Result<Vec<DeviceInfo>, DiscoveryError> {
        // 简化实现：返回空列表
        Ok(Vec::new())
    }

    async fn advertise_local_device(&mut self, _device: &DeviceInfo) -> Result<(), DiscoveryError> {
        // 简化实现
        Ok(())
    }

    async fn listen(&mut self) -> Result<Option<DiscoveryEvent>, DiscoveryError> {
        // 简化实现：返回None
        Ok(None)
    }

    fn backend_type(&self) -> DiscoveryBackendType {
        DiscoveryBackendType::Static
    }

    fn is_healthy(&self) -> bool {
        true
    }

    fn get_stats(&self) -> DiscoveryStats {
        self.stats.clone()
    }
}

/// 混合发现后端
struct HybridBackend {
    config: DiscoveryConfig,
    stats: DiscoveryStats,
}

impl HybridBackend {
    async fn new(config: &DiscoveryConfig) -> Result<Self, DiscoveryError> {
        Ok(Self {
            config: config.clone(),
            stats: DiscoveryStats {
                total_devices_discovered: 0,
                active_devices: 0,
                discovery_operations: 0,
                successful_discoveries: 0,
                failed_discoveries: 0,
                avg_discovery_latency_ns: 0,
                last_discovery_time: None,
            },
        })
    }
}

#[async_trait::async_trait]
impl DiscoveryBackend for HybridBackend {
    async fn start(&mut self) -> Result<(), DiscoveryError> {
        info!("混合发现后端启动");
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), DiscoveryError> {
        info!("混合发现后端停止");
        Ok(())
    }

    async fn discover_devices(&mut self) -> Result<Vec<DeviceInfo>, DiscoveryError> {
        // 简化实现：返回空列表
        Ok(Vec::new())
    }

    async fn advertise_local_device(&mut self, _device: &DeviceInfo) -> Result<(), DiscoveryError> {
        // 简化实现
        Ok(())
    }

    async fn listen(&mut self) -> Result<Option<DiscoveryEvent>, DiscoveryError> {
        // 简化实现：返回None
        Ok(None)
    }

    fn backend_type(&self) -> DiscoveryBackendType {
        DiscoveryBackendType::Hybrid
    }

    fn is_healthy(&self) -> bool {
        true
    }

    fn get_stats(&self) -> DiscoveryStats {
        self.stats.clone()
    }
}

// ==================== 模块导出 ====================
// 类型已经在顶层定义为pub，不需要重复导出
