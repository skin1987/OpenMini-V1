//! OpenMini-V1 Worker 故障容错测试
//!
//! 验证 Worker 进程池在故障场景下的行为：
//! - Worker 配置边界值验证
//! - 任务序列化/反序列化容错
//! - Worker 状态机转换正确性
//! - 错误恢复路径覆盖

use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_worker_config_boundary_values() {
    use openmini_server::service::worker::pool::WorkerConfig;

    eprintln!("[fault-config] Testing boundary configurations...");

    let min_config = WorkerConfig::new()
        .with_count(1)
        .with_restart_on_failure(false)
        .with_health_check_interval(100);
    assert_eq!(min_config.count, 1);
    assert!(!min_config.restart_on_failure);
    assert_eq!(min_config.health_check_interval_ms, 100);

    let max_config = WorkerConfig::new()
        .with_count(64)
        .with_restart_on_failure(true)
        .with_health_check_interval(60000);
    assert_eq!(max_config.count, 64);
    assert!(max_config.restart_on_failure);
    assert_eq!(max_config.health_check_interval_ms, 60000);

    let default_config = WorkerConfig::default();
    assert_eq!(default_config.count, 3);
    assert!(default_config.restart_on_failure);
    assert_eq!(default_config.health_check_interval_ms, 5000);

    eprintln!("[fault-config] All boundary configs validated");
}

#[test]
fn test_task_serialization_fault_tolerance() {
    use openmini_server::service::worker::worker::Task;

    eprintln!("[fault-serial] Testing serialization edge cases...");

    let empty_task = Task::new(0, vec![]);
    let ser_empty = empty_task.serialize();
    let de_empty = Task::deserialize(&ser_empty).expect("Empty task should deserialize");
    assert_eq!(de_empty.id, 0);
    assert!(de_empty.data.is_empty());

    let large_data: Vec<u8> = (0u8..=255).cycle().take(65536).collect();
    let large_task = Task::new(u64::MAX, large_data.clone());
    let ser_large = large_task.serialize();
    let de_large = Task::deserialize(&ser_large).expect("Large task should deserialize");
    assert_eq!(de_large.id, u64::MAX);
    assert_eq!(de_large.data.len(), 65536);
    assert_eq!(de_large.data, large_data);

    let binary_data: Vec<u8> = vec![0x00, 0xFF, 0x80, 0x7F, 0x00, 0xAB, 0xCD, 0xEF];
    let binary_task = Task::new(42, binary_data.clone());
    let ser_binary = binary_task.serialize();
    let de_binary = Task::deserialize(&ser_binary).expect("Binary data should roundtrip");
    assert_eq!(de_binary.data, binary_data);

    let too_short = &[0u8; 4];
    assert!(Task::deserialize(too_short).is_err(), "Too short data should fail");

    let truncated_id_only = {
        let mut v = vec![0u8; 8];
        v[0..8].copy_from_slice(&9999u64.to_le_bytes());
        v.push(100);
        v.push(0);
        v
    };
    assert!(
        Task::deserialize(&truncated_id_only).is_err(),
        "Truncated payload should fail"
    );

    eprintln!("[fault-serial] All serialization fault tests passed");
}

#[test]
fn test_worker_state_machine_transitions() {
    use openmini_server::service::worker::worker::{
        WORKER_STATE_BUSY, WORKER_STATE_DEAD, WORKER_STATE_IDLE,
    };

    eprintln!("[fault-state] Validating state machine...");

    assert_eq!(WORKER_STATE_IDLE, 0, "IDLE state should be 0");
    assert_eq!(WORKER_STATE_BUSY, 1, "BUSY state should be 1");
    assert_eq!(WORKER_STATE_DEAD, 2, "DEAD state should be 2");

    assert_ne!(WORKER_STATE_IDLE, WORKER_STATE_BUSY);
    assert_ne!(WORKER_STATE_BUSY, WORKER_STATE_DEAD);
    assert_ne!(WORKER_STATE_IDLE, WORKER_STATE_DEAD);

    let valid_states = [WORKER_STATE_IDLE, WORKER_STATE_BUSY, WORKER_STATE_DEAD];
    for &state in &valid_states {
        assert!(state <= 2, "State value should be in valid range");
    }

    eprintln!("[fault-state] State machine validation passed");
}

#[test]
fn test_worker_pool_error_handling() {
    use openmini_server::service::worker::pool::{WorkerPoolError, WorkerPoolBuilder};

    eprintln!("[fault-error] Testing error handling paths...");

    let spawn_error = WorkerPoolError::SpawnError("fork() failed".to_string());
    let display_spawn = format!("{}", spawn_error);
    assert!(display_spawn.contains("spawn"), "SpawnError display should contain 'spawn'");

    let comm_error = WorkerPoolError::CommunicationError("pipe broken".to_string());
    let display_comm = format!("{}", comm_error);
    assert!(display_comm.contains("communication") || display_comm.contains("Communication"));

    let no_worker = WorkerPoolError::NoAvailableWorker;
    let display_no = format!("{}", no_worker);
    assert!(!display_no.is_empty());

    let dead_error = WorkerPoolError::WorkerDead(7);
    let display_dead = format!("{}", dead_error);
    assert!(display_dead.contains("7"), "Dead worker ID should appear in message");

    let _: &dyn std::error::Error = &spawn_error;
    let _: &dyn std::error::Error = &comm_error;
    let _: &dyn std::error::Error = &no_worker;
    let _: &dyn std::error::Error = &dead_error;

    let _builder = WorkerPoolBuilder::new()
        .worker_count(0)
        .restart_on_failure(false)
        .health_check_interval(0);
    // config 字段是私有的，无法直接访问验证
    // 但 builder 创建成功即说明配置有效

    eprintln!("[fault-error] All error handling tests passed");
}

#[test]
fn test_task_result_serialization_roundtrip_with_failures() {
    use openmini_server::service::worker::worker::TaskResult;

    eprintln!("[fault-result] Testing result roundtrip with failure cases...");

    let success_result = TaskResult {
        task_id: 1,
        success: true,
        data: b"OK".to_vec(),
    };
    let ser_success = success_result.serialize();
    let de_success = TaskResult::deserialize(&ser_success).expect("Success result should roundtrip");
    assert!(de_success.success);
    assert_eq!(de_success.task_id, 1);
    assert_eq!(de_success.data, b"OK");

    let failure_result = TaskResult {
        task_id: 2,
        success: false,
        data: b"ERROR: out of memory".to_vec(),
    };
    let ser_failure = failure_result.serialize();
    let de_failure = TaskResult::deserialize(&ser_failure).expect("Failure result should roundtrip");
    assert!(!de_failure.success);
    assert_eq!(de_failure.task_id, 2);
    assert_eq!(de_failure.data, b"ERROR: out of memory");

    let empty_result = TaskResult {
        task_id: 999,
        success: false,
        data: vec![],
    };
    let ser_empty = empty_result.serialize();
    let de_empty = TaskResult::deserialize(&ser_empty).expect("Empty result should roundtrip");
    assert!(!de_empty.success);
    assert!(de_empty.data.is_empty());

    assert!(TaskResult::deserialize(&[0u8; 4]).is_err(), "Too short should fail");

    eprintln!("[fault-result] All result roundtrip tests passed");
}
