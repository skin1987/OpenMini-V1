//! 连接池 - TCP 连接复用和管理
//!
//! 提供高效的 TCP 连接池实现，支持:
//! - 连接复用
//! - 连接超时管理
//! - 连接统计
//! - 过期连接清理

#![allow(dead_code)]

use parking_lot::RwLock;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, ToSocketAddrs};
use tokio::sync::Semaphore;
use tokio::time::timeout;

/// 默认连接超时时间
const DEFAULT_CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);
/// 默认空闲超时时间
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
/// 最大重试次数
#[allow(dead_code)]
const MAX_RETRIES: usize = 3;

/// 连接状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// 空闲状态
    Idle,
    /// 活跃状态
    Active,
    /// 已关闭
    Closed,
}

/// 连接统计信息
///
/// 使用原子计数器实现线程安全的统计。
pub struct ConnectionStats {
    /// 总创建连接数
    pub total_created: AtomicU64,
    /// 总复用连接数
    pub total_reused: AtomicU64,
    /// 总关闭连接数
    pub total_closed: AtomicU64,
    /// 当前活跃连接数
    pub active_connections: AtomicUsize,
    /// 获取连接次数
    pub acquire_count: AtomicU64,
    /// 释放连接次数
    pub release_count: AtomicU64,
}

impl ConnectionStats {
    /// 创建新的统计实例
    pub fn new() -> Self {
        Self {
            total_created: AtomicU64::new(0),
            total_reused: AtomicU64::new(0),
            total_closed: AtomicU64::new(0),
            active_connections: AtomicUsize::new(0),
            acquire_count: AtomicU64::new(0),
            release_count: AtomicU64::new(0),
        }
    }

    /// 计算连接复用率
    pub fn reuse_rate(&self) -> f64 {
        let total =
            self.total_created.load(Ordering::Relaxed) + self.total_reused.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.total_reused.load(Ordering::Relaxed) as f64 / total as f64
        }
    }

    /// 获取统计快照
    pub fn snapshot(&self) -> ConnectionStatsSnapshot {
        ConnectionStatsSnapshot {
            total_created: self.total_created.load(Ordering::Relaxed),
            total_reused: self.total_reused.load(Ordering::Relaxed),
            total_closed: self.total_closed.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            reuse_rate: self.reuse_rate(),
        }
    }
}

impl Default for ConnectionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// 连接统计快照
#[derive(Debug, Clone)]
pub struct ConnectionStatsSnapshot {
    /// 总创建连接数
    pub total_created: u64,
    /// 总复用连接数
    pub total_reused: u64,
    /// 总关闭连接数
    pub total_closed: u64,
    /// 当前活跃连接数
    pub active_connections: usize,
    /// 复用率
    pub reuse_rate: f64,
}

/// TCP 连接封装
///
/// 封装 TcpStream，提供状态管理和超时检测。
pub struct Connection {
    /// TCP 流
    stream: Option<TcpStream>,
    /// 当前状态
    state: ConnectionState,
    /// 创建时间
    created_at: Instant,
    /// 最后使用时间
    last_used: Instant,
    /// 远程地址
    remote_addr: SocketAddr,
    /// 连接 ID
    id: u64,
}

impl Connection {
    /// 创建新连接
    fn new(stream: TcpStream, remote_addr: SocketAddr, id: u64) -> Self {
        Self {
            stream: Some(stream),
            state: ConnectionState::Idle,
            created_at: Instant::now(),
            last_used: Instant::now(),
            remote_addr,
            id,
        }
    }

    /// 获取连接 ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 获取当前状态
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// 获取远程地址
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// 获取连接存活时间
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// 获取空闲时间
    pub fn idle_duration(&self) -> Duration {
        self.last_used.elapsed()
    }

    /// 检查是否已过期
    pub fn is_expired(&self, max_idle: Duration) -> bool {
        self.idle_duration() > max_idle
    }

    /// 标记为活跃状态
    fn mark_active(&mut self) {
        self.state = ConnectionState::Active;
        self.last_used = Instant::now();
    }

    /// 标记为空闲状态
    fn mark_idle(&mut self) {
        self.state = ConnectionState::Idle;
        self.last_used = Instant::now();
    }

    /// 标记为已关闭
    fn mark_closed(&mut self) {
        self.state = ConnectionState::Closed;
    }

    /// 异步读取数据
    pub async fn read(&mut self, buf: &mut [u8]) -> tokio::io::Result<usize> {
        if let Some(ref mut stream) = self.stream {
            stream.read(buf).await
        } else {
            Err(tokio::io::Error::new(
                tokio::io::ErrorKind::NotConnected,
                "Connection stream is None",
            ))
        }
    }

    /// 异步精确读取指定长度
    pub async fn read_exact(&mut self, buf: &mut [u8]) -> tokio::io::Result<usize> {
        if let Some(ref mut stream) = self.stream {
            stream.read_exact(buf).await?;
            Ok(buf.len())
        } else {
            Err(tokio::io::Error::new(
                tokio::io::ErrorKind::NotConnected,
                "Connection stream is None",
            ))
        }
    }

    /// 异步写入数据
    pub async fn write(&mut self, buf: &[u8]) -> tokio::io::Result<usize> {
        if let Some(ref mut stream) = self.stream {
            stream.write(buf).await
        } else {
            Err(tokio::io::Error::new(
                tokio::io::ErrorKind::NotConnected,
                "Connection stream is None",
            ))
        }
    }

    /// 异步写入全部数据
    pub async fn write_all(&mut self, buf: &[u8]) -> tokio::io::Result<()> {
        if let Some(ref mut stream) = self.stream {
            stream.write_all(buf).await
        } else {
            Err(tokio::io::Error::new(
                tokio::io::ErrorKind::NotConnected,
                "Connection stream is None",
            ))
        }
    }

    /// 刷新写入缓冲区
    pub async fn flush(&mut self) -> tokio::io::Result<()> {
        if let Some(ref mut stream) = self.stream {
            stream.flush().await
        } else {
            Ok(())
        }
    }

    /// 关闭连接
    pub async fn shutdown(&mut self) -> tokio::io::Result<()> {
        if let Some(ref mut stream) = self.stream {
            stream.shutdown().await?;
        }
        self.mark_closed();
        Ok(())
    }
}

/// 连接池
///
/// 管理可复用的 TCP 连接，支持连接限制和超时管理。
pub struct ConnectionPool {
    /// 最大连接数
    max_connections: usize,
    /// 空闲连接队列
    connections: RwLock<VecDeque<Connection>>,
    /// 连接信号量
    semaphore: Arc<Semaphore>,
    /// 统计信息
    stats: Arc<ConnectionStats>,
    /// 下一个连接 ID
    next_connection_id: AtomicU64,
    /// 空闲超时
    idle_timeout: Duration,
    /// 连接超时
    connection_timeout: Duration,
}

impl ConnectionPool {
    /// 创建新的连接池
    pub fn new(max_connections: usize) -> Self {
        Self::with_options(
            max_connections,
            DEFAULT_IDLE_TIMEOUT,
            DEFAULT_CONNECTION_TIMEOUT,
        )
    }

    /// 使用自定义选项创建连接池
    pub fn with_options(
        max_connections: usize,
        idle_timeout: Duration,
        connection_timeout: Duration,
    ) -> Self {
        Self {
            max_connections,
            connections: RwLock::new(VecDeque::with_capacity(max_connections)),
            semaphore: Arc::new(Semaphore::new(max_connections)),
            stats: Arc::new(ConnectionStats::new()),
            next_connection_id: AtomicU64::new(1),
            idle_timeout,
            connection_timeout,
        }
    }

    /// 建立新连接
    pub async fn connect<A: ToSocketAddrs + std::fmt::Debug + Clone + Send + 'static>(
        &self,
        addr: A,
    ) -> tokio::io::Result<Connection> {
        let permit = self.semaphore.acquire().await.map_err(|_| {
            tokio::io::Error::new(
                tokio::io::ErrorKind::WouldBlock,
                "Connection pool exhausted",
            )
        })?;

        let addr_clone = addr.clone();
        let connect_result = timeout(self.connection_timeout, async {
            TcpStream::connect(&addr_clone).await
        })
        .await;

        match connect_result {
            Ok(Ok(stream)) => {
                let remote_addr = stream.peer_addr()?;
                let id = self.next_connection_id.fetch_add(1, Ordering::Relaxed);
                let mut conn = Connection::new(stream, remote_addr, id);
                conn.mark_active();

                self.stats.total_created.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .active_connections
                    .fetch_add(1, Ordering::Relaxed);
                self.stats.acquire_count.fetch_add(1, Ordering::Relaxed);

                permit.forget();
                Ok(conn)
            }
            Ok(Err(e)) => {
                drop(permit);
                Err(e)
            }
            Err(_) => {
                drop(permit);
                Err(tokio::io::Error::new(
                    tokio::io::ErrorKind::TimedOut,
                    "Connection timeout",
                ))
            }
        }
    }

    /// 从池中获取空闲连接
    pub async fn acquire(&self) -> Option<Connection> {
        let permit = self.semaphore.acquire().await.ok()?;

        let mut connections = self.connections.write();

        while let Some(mut conn) = connections.pop_front() {
            if conn.is_expired(self.idle_timeout) {
                self.stats.total_closed.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            conn.mark_active();
            self.stats.total_reused.fetch_add(1, Ordering::Relaxed);
            self.stats
                .active_connections
                .fetch_add(1, Ordering::Relaxed);
            self.stats.acquire_count.fetch_add(1, Ordering::Relaxed);
            permit.forget();
            return Some(conn);
        }

        drop(connections);
        self.semaphore.add_permits(1);
        None
    }

    /// 释放连接回池
    pub fn release(&self, mut conn: Connection) {
        if conn.state == ConnectionState::Closed || conn.is_expired(self.idle_timeout) {
            self.stats.total_closed.fetch_add(1, Ordering::Relaxed);
            self.stats
                .active_connections
                .fetch_sub(1, Ordering::Relaxed);
            self.stats.release_count.fetch_add(1, Ordering::Relaxed);
            self.semaphore.add_permits(1);
            return;
        }

        conn.mark_idle();

        let mut connections = self.connections.write();
        if connections.len() < self.max_connections {
            connections.push_back(conn);
            self.stats
                .active_connections
                .fetch_sub(1, Ordering::Relaxed);
            self.stats.release_count.fetch_add(1, Ordering::Relaxed);
            self.semaphore.add_permits(1);
        } else {
            self.stats.total_closed.fetch_add(1, Ordering::Relaxed);
            self.stats
                .active_connections
                .fetch_sub(1, Ordering::Relaxed);
            self.stats.release_count.fetch_add(1, Ordering::Relaxed);
            self.semaphore.add_permits(1);
        }
    }

    /// 获取或创建连接
    pub async fn acquire_or_connect<A: ToSocketAddrs + std::fmt::Debug + Clone + Send + 'static>(
        &self,
        addr: A,
    ) -> tokio::io::Result<Connection> {
        if let Some(conn) = self.acquire().await {
            return Ok(conn);
        }
        self.connect(addr).await
    }

    /// 获取统计信息
    pub fn stats(&self) -> Arc<ConnectionStats> {
        Arc::clone(&self.stats)
    }

    /// 获取可用连接数
    pub fn available(&self) -> usize {
        self.connections.read().len()
    }

    /// 获取最大连接数
    pub fn max_connections(&self) -> usize {
        self.max_connections
    }

    /// 清理过期连接
    pub async fn cleanup_expired(&self) {
        let mut connections = self.connections.write();
        let before = connections.len();
        connections.retain(|conn| !conn.is_expired(self.idle_timeout));
        let removed = before - connections.len();

        if removed > 0 {
            self.stats
                .total_closed
                .fetch_add(removed as u64, Ordering::Relaxed);
            self.semaphore.add_permits(removed);
        }
    }

    /// 关闭所有连接
    pub async fn close_all(&self) {
        let mut connections = self.connections.write();
        let count = connections.len();
        connections.clear();

        self.stats
            .total_closed
            .fetch_add(count as u64, Ordering::Relaxed);
        self.semaphore.add_permits(count);
    }
}

/// 池化连接守卫
///
/// 自动在 drop 时将连接归还到池中。
pub struct PooledConnection<'a> {
    connection: Option<Connection>,
    pool: &'a ConnectionPool,
}

impl<'a> PooledConnection<'a> {
    /// 创建新的池化连接
    pub fn new(connection: Connection, pool: &'a ConnectionPool) -> Self {
        Self {
            connection: Some(connection),
            pool,
        }
    }

    /// 获取连接引用
    pub fn connection(&self) -> Option<&Connection> {
        self.connection.as_ref()
    }

    /// 获取可变连接引用
    pub fn connection_mut(&mut self) -> Option<&mut Connection> {
        self.connection.as_mut()
    }
}

impl<'a> Drop for PooledConnection<'a> {
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            self.pool.release(conn);
        }
    }
}

/// 连接池构建器
pub struct ConnectionBuilder {
    max_connections: usize,
    idle_timeout: Duration,
    connection_timeout: Duration,
}

impl ConnectionBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            max_connections: 100,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            connection_timeout: DEFAULT_CONNECTION_TIMEOUT,
        }
    }

    /// 设置最大连接数
    pub fn max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    /// 设置空闲超时
    pub fn idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// 设置连接超时
    pub fn connection_timeout(mut self, timeout: Duration) -> Self {
        self.connection_timeout = timeout;
        self
    }

    /// 构建连接池
    pub fn build(self) -> ConnectionPool {
        ConnectionPool::with_options(
            self.max_connections,
            self.idle_timeout,
            self.connection_timeout,
        )
    }
}

impl Default for ConnectionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn test_connection_pool_new() {
        let pool = ConnectionPool::new(10);
        assert_eq!(pool.max_connections(), 10);
        assert_eq!(pool.available(), 0);
    }

    #[tokio::test]
    async fn test_connection_stats() {
        let stats = ConnectionStats::new();
        assert_eq!(stats.total_created.load(Ordering::Relaxed), 0);
        assert_eq!(stats.reuse_rate(), 0.0);
    }

    #[tokio::test]
    async fn test_connection_builder() {
        let pool = ConnectionBuilder::new()
            .max_connections(50)
            .idle_timeout(Duration::from_secs(60))
            .connection_timeout(Duration::from_secs(10))
            .build();

        assert_eq!(pool.max_connections(), 50);
    }

    #[tokio::test]
    async fn test_connection_pool_acquire_release() {
        let pool = ConnectionPool::new(2);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let _ = stream;
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(10)).await;

        let conn = pool.connect(addr).await.unwrap();
        assert_eq!(conn.state(), ConnectionState::Active);

        pool.release(conn);
        assert_eq!(pool.available(), 1);

        let conn = pool.acquire().await.unwrap();
        assert_eq!(conn.state(), ConnectionState::Active);

        pool.release(conn);

        server_task.abort();
    }

    #[tokio::test]
    async fn test_connection_pool_basic_operations() {
        // 基本连接池操作
        let pool = ConnectionPool::new(5);

        // 初始状态验证
        assert_eq!(pool.max_connections(), 5);
        assert_eq!(pool.available(), 0); // 新池子没有空闲连接

        // 验证统计信息初始状态
        let stats = pool.stats();
        assert_eq!(stats.total_created.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_reused.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_closed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.active_connections.load(Ordering::Relaxed), 0);
        assert_eq!(stats.reuse_rate(), 0.0);
    }

    #[tokio::test]
    async fn test_connection_pool_connect_and_release_cycle() {
        // 连接获取和释放循环
        let pool = ConnectionPool::new(3);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let _ = stream;
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(10)).await;

        // 获取连接1
        let conn1 = pool.connect(addr).await.unwrap();
        assert_eq!(conn1.state(), ConnectionState::Active);
        assert_eq!(pool.available(), 0);

        // 获取连接2
        let conn2 = pool.connect(addr).await.unwrap();
        assert_eq!(conn2.state(), ConnectionState::Active);
        assert_eq!(pool.available(), 0);

        // 释放连接1
        pool.release(conn1);
        assert_eq!(pool.available(), 1);

        // 从池中获取(应该复用)
        let conn3 = pool.acquire().await.unwrap();
        assert_eq!(conn3.state(), ConnectionState::Active);
        assert_eq!(pool.available(), 0);

        // 验证复用统计
        let stats = pool.stats();
        assert_eq!(stats.total_created.load(Ordering::Relaxed), 2); // 创建了2个新连接
        assert_eq!(stats.total_reused.load(Ordering::Relaxed), 1); // 复用了1个
        assert!(stats.reuse_rate() > 0.0);

        // 释放剩余连接
        pool.release(conn2);
        pool.release(conn3);

        server_task.abort();
    }

    #[tokio::test]
    async fn test_connection_pool_exhaustion() {
        // 连接池耗尽测试
        let pool = ConnectionPool::new(2);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let _ = stream;
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(10)).await;

        // 获取所有连接
        let conn1 = pool.connect(addr).await.unwrap();
        let conn2 = pool.connect(addr).await.unwrap();

        // 尝试获取第3个连接 - acquire应该返回None(因为没有空闲连接)
        let conn3 = pool.acquire().await;
        assert!(
            conn3.is_none(),
            "Should return None when no connections available"
        );

        // 释放一个连接
        pool.release(conn1);

        // 现在应该能获取到
        let conn4 = pool.acquire().await;
        assert!(conn4.is_some(), "Should get connection after release");

        // 清理
        pool.release(conn2);
        if let Some(c) = conn4 {
            pool.release(c);
        }

        server_task.abort();
    }

    #[tokio::test]
    async fn test_connection_idle_timeout() {
        // 空闲超时测试
        let pool = ConnectionPool::with_options(
            5,
            Duration::from_millis(100), // 100ms空闲超时
            Duration::from_secs(30),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let _ = stream;
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(10)).await;

        // 创建并释放连接
        let conn = pool.connect(addr).await.unwrap();
        let _conn_id = conn.id();
        pool.release(conn);

        assert_eq!(pool.available(), 1);

        // 等待超时
        tokio::time::sleep(Duration::from_millis(150)).await;

        // 清理过期连接
        pool.cleanup_expired().await;

        // 连接应该被清理了
        assert_eq!(
            pool.available(),
            0,
            "Expired connection should be cleaned up"
        );

        // 验证关闭统计
        let stats = pool.stats();
        assert_eq!(stats.total_closed.load(Ordering::Relaxed), 1);

        server_task.abort();
    }

    #[tokio::test]
    async fn test_connection_stats_tracking() {
        // 连接统计跟踪测试
        let pool = ConnectionPool::new(3);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let _ = stream;
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(10)).await;

        // 创建连接
        let conn1 = pool.connect(addr).await.unwrap();
        let stats = pool.stats().snapshot();
        assert_eq!(stats.total_created, 1);
        assert_eq!(stats.active_connections, 1);

        // 再创建一个
        let conn2 = pool.connect(addr).await.unwrap();
        let stats = pool.stats().snapshot();
        assert_eq!(stats.total_created, 2);
        assert_eq!(stats.active_connections, 2);

        // 释放第一个
        pool.release(conn1);
        let stats = pool.stats().snapshot();
        assert_eq!(stats.active_connections, 1);

        // 复用
        let _conn3 = pool.acquire().await.unwrap();
        let stats = pool.stats().snapshot();
        assert_eq!(stats.total_reused, 1);
        assert_eq!(stats.active_connections, 2);

        // 清理
        pool.release(conn2);
        // 注意: conn3 已经被移动,不需要再次release

        server_task.abort();
    }

    #[tokio::test]
    async fn test_connection_builder_pattern() {
        // 构建器模式测试
        let pool = ConnectionBuilder::new()
            .max_connections(50)
            .idle_timeout(Duration::from_secs(60))
            .connection_timeout(Duration::from_secs(10))
            .build();

        assert_eq!(pool.max_connections(), 50);

        // 测试默认构建器
        let default_pool = ConnectionBuilder::new().build();
        assert_eq!(default_pool.max_connections(), 100); // 默认值
    }

    #[tokio::test]
    async fn test_connection_state_transitions() {
        // 连接状态转换测试
        let pool = ConnectionPool::new(2);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let _ = stream;
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(10)).await;

        // 创建连接 - 应该是Active状态
        let mut conn = pool.connect(addr).await.unwrap();
        assert_eq!(conn.state(), ConnectionState::Active);

        // 释放后变为Idle
        pool.release(conn);

        // 再次获取 - 变为Active
        conn = pool.acquire().await.unwrap();
        assert_eq!(conn.state(), ConnectionState::Active);

        // 关闭连接
        conn.shutdown().await.unwrap();
        assert_eq!(conn.state(), ConnectionState::Closed);

        // 释放已关闭的连接 - 应该被清理而不是放回池中
        pool.release(conn);
        assert_eq!(
            pool.available(),
            0,
            "Closed connection should not be returned to pool"
        );

        server_task.abort();
    }

    #[tokio::test]
    async fn test_pooled_connection_guard() {
        // 池化连接守卫测试
        let pool = Arc::new(ConnectionPool::new(2));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let _ = stream;
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(10)).await;

        // 创建连接并用守卫包装
        let conn = pool.connect(addr).await.unwrap();
        let guard = PooledConnection::new(conn, &pool);

        // 通过守卫访问连接
        assert!(guard.connection().is_some());
        assert_eq!(guard.connection().unwrap().state(), ConnectionState::Active);

        // drop守卫应该自动归还连接
        drop(guard);

        // 连接应该回到池中
        assert_eq!(
            pool.available(),
            1,
            "Connection should be returned to pool after guard drop"
        );

        server_task.abort();
    }

    #[tokio::test]
    async fn test_connection_close_all() {
        // 关闭所有连接测试
        let pool = ConnectionPool::new(3);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let _ = stream;
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(10)).await;

        // 创建多个连接
        let c1 = pool.connect(addr).await.unwrap();
        let c2 = pool.connect(addr).await.unwrap();
        let c3 = pool.connect(addr).await.unwrap();

        // 释放所有连接到池中
        pool.release(c1);
        pool.release(c2);
        pool.release(c3);

        assert_eq!(pool.available(), 3);

        // 关闭所有连接
        pool.close_all().await;

        assert_eq!(pool.available(), 0, "All connections should be closed");

        // 验证关闭统计
        let stats = pool.stats();
        assert_eq!(stats.total_closed.load(Ordering::Relaxed), 3);

        server_task.abort();
    }
}
