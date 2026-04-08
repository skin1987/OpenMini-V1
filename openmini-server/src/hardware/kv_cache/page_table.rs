//! 页表实现
//!
//! 管理逻辑地址到物理Block的映射

use super::block::BlockId;

/// 页表项
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct PageTableEntry {
    /// 物理块ID
    pub block_id: BlockId,
    /// 是否有效
    pub valid: bool,
}

#[allow(dead_code)]
impl PageTableEntry {
    /// 创建新的页表项
    pub fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            valid: true,
        }
    }

    /// 创建无效的页表项
    pub fn invalid() -> Self {
        Self {
            block_id: 0,
            valid: false,
        }
    }
}

/// 页表 - 管理逻辑地址到物理Block的映射
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PageTable {
    /// 逻辑到物理的映射
    entries: Vec<PageTableEntry>,
    /// 当前使用的槽位数
    num_slots: usize,
    /// 最大槽位数
    max_slots: usize,
}

#[allow(dead_code)]
impl PageTable {
    /// 创建新的页表
    pub fn new(max_slots: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_slots),
            num_slots: 0,
            max_slots,
        }
    }

    /// 从块ID列表创建页表
    pub fn from_blocks(block_ids: Vec<BlockId>) -> Self {
        let entries: Vec<PageTableEntry> = block_ids
            .into_iter()
            .map(PageTableEntry::new)
            .collect();
        let num_slots = entries.len();
        
        Self {
            max_slots: num_slots,
            num_slots,
            entries,
        }
    }

    /// 追加块
    pub fn append(&mut self, block_id: BlockId) -> Result<(), String> {
        if self.num_slots >= self.max_slots {
            return Err("Page table is full".to_string());
        }
        
        self.entries.push(PageTableEntry::new(block_id));
        self.num_slots += 1;
        Ok(())
    }

    /// 追加多个块
    pub fn append_blocks(&mut self, block_ids: Vec<BlockId>) -> Result<(), String> {
        let new_slots = block_ids.len();
        if self.num_slots + new_slots > self.max_slots {
            return Err(format!(
                "Page table overflow: current {}, adding {}, max {}",
                self.num_slots, new_slots, self.max_slots
            ));
        }
        
        for block_id in block_ids {
            self.entries.push(PageTableEntry::new(block_id));
            self.num_slots += 1;
        }
        Ok(())
    }

    /// 获取指定位置的块ID
    pub fn get(&self, slot: usize) -> Option<BlockId> {
        self.entries.get(slot).and_then(|entry| {
            if entry.valid {
                Some(entry.block_id)
            } else {
                None
            }
        })
    }

    /// 获取所有有效的块ID
    pub fn get_all_blocks(&self) -> Vec<BlockId> {
        self.entries
            .iter()
            .filter(|entry| entry.valid)
            .map(|entry| entry.block_id)
            .collect()
    }

    /// 获取最后一个块ID
    pub fn last_block(&self) -> Option<BlockId> {
        self.entries.iter().rev().find(|entry| entry.valid).map(|entry| entry.block_id)
    }

    /// 设置指定位置的块ID
    pub fn set(&mut self, slot: usize, block_id: BlockId) -> Result<(), String> {
        if slot >= self.num_slots {
            return Err(format!("Slot {} out of range {}", slot, self.num_slots));
        }
        
        self.entries[slot] = PageTableEntry::new(block_id);
        Ok(())
    }

    /// 使指定位置无效
    pub fn invalidate(&mut self, slot: usize) {
        if slot < self.entries.len() {
            self.entries[slot].valid = false;
        }
    }

    /// 弹出最后一个块
    pub fn pop(&mut self) -> Option<BlockId> {
        if let Some(entry) = self.entries.pop() {
            self.num_slots = self.num_slots.saturating_sub(1);
            if entry.valid {
                Some(entry.block_id)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// 清空页表
    pub fn clear(&mut self) {
        self.entries.clear();
        self.num_slots = 0;
    }

    /// 获取当前槽位数
    pub fn len(&self) -> usize {
        self.num_slots
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.num_slots == 0
    }

    /// 获取最大槽位数
    pub fn capacity(&self) -> usize {
        self.max_slots
    }

    /// 扩展最大槽位数
    pub fn reserve(&mut self, additional: usize) {
        self.max_slots += additional;
        self.entries.reserve(additional);
    }

    /// 获取有效条目数
    pub fn valid_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.valid).count()
    }

    /// 检查槽位是否有效
    pub fn is_valid(&self, slot: usize) -> bool {
        self.entries.get(slot).map(|entry| entry.valid).unwrap_or(false)
    }

    /// 截断到指定长度
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.num_slots {
            self.entries.truncate(new_len);
            self.num_slots = new_len;
        }
    }

    /// 克隆页表（深拷贝）
    pub fn fork(&self) -> Self {
        self.clone()
    }
}

impl Default for PageTable {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// 页表管理器 - 管理多个请求的页表
#[derive(Debug)]
#[allow(dead_code)]
pub struct PageTableManager {
    /// 请求ID到页表的映射
    page_tables: std::collections::HashMap<u64, PageTable>,
    /// 下一个请求ID
    next_request_id: std::sync::atomic::AtomicU64,
}

#[allow(dead_code)]
impl PageTableManager {
    /// 创建新的页表管理器
    pub fn new() -> Self {
        Self {
            page_tables: std::collections::HashMap::new(),
            next_request_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// 创建新的页表
    pub fn create_page_table(&mut self, max_slots: usize) -> u64 {
        let request_id = self.next_request_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.page_tables.insert(request_id, PageTable::new(max_slots));
        request_id
    }

    /// 获取页表
    pub fn get_page_table(&self, request_id: u64) -> Option<&PageTable> {
        self.page_tables.get(&request_id)
    }

    /// 获取可变页表
    pub fn get_page_table_mut(&mut self, request_id: u64) -> Option<&mut PageTable> {
        self.page_tables.get_mut(&request_id)
    }

    /// 删除页表
    pub fn remove_page_table(&mut self, request_id: u64) -> Option<PageTable> {
        self.page_tables.remove(&request_id)
    }

    /// 检查页表是否存在
    pub fn contains(&self, request_id: u64) -> bool {
        self.page_tables.contains_key(&request_id)
    }

    /// 获取页表数量
    pub fn len(&self) -> usize {
        self.page_tables.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.page_tables.is_empty()
    }

    /// 获取所有请求ID
    pub fn request_ids(&self) -> Vec<u64> {
        self.page_tables.keys().copied().collect()
    }

    /// 清空所有页表
    pub fn clear(&mut self) {
        self.page_tables.clear();
    }
}

impl Default for PageTableManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_table_new() {
        let pt = PageTable::new(100);
        assert_eq!(pt.len(), 0);
        assert!(pt.is_empty());
        assert_eq!(pt.capacity(), 100);
    }

    #[test]
    fn test_page_table_from_blocks() {
        let pt = PageTable::from_blocks(vec![0, 1, 2, 3]);
        assert_eq!(pt.len(), 4);
        assert!(!pt.is_empty());
        assert_eq!(pt.get(0), Some(0));
        assert_eq!(pt.get(3), Some(3));
        assert_eq!(pt.get(4), None);
    }

    #[test]
    fn test_page_table_append() {
        let mut pt = PageTable::new(10);
        
        pt.append(0).unwrap();
        pt.append(1).unwrap();
        pt.append(2).unwrap();
        
        assert_eq!(pt.len(), 3);
        assert_eq!(pt.get(0), Some(0));
        assert_eq!(pt.get(1), Some(1));
        assert_eq!(pt.get(2), Some(2));
    }

    #[test]
    fn test_page_table_overflow() {
        let mut pt = PageTable::new(2);
        
        pt.append(0).unwrap();
        pt.append(1).unwrap();
        
        assert!(pt.append(2).is_err());
        assert_eq!(pt.len(), 2);
    }

    #[test]
    fn test_page_table_invalidate() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        
        pt.invalidate(1);
        
        assert_eq!(pt.get(0), Some(0));
        assert_eq!(pt.get(1), None);
        assert_eq!(pt.get(2), Some(2));
        assert_eq!(pt.valid_count(), 2);
    }

    #[test]
    fn test_page_table_pop() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        
        assert_eq!(pt.pop(), Some(2));
        assert_eq!(pt.len(), 2);
        assert_eq!(pt.pop(), Some(1));
        assert_eq!(pt.len(), 1);
    }

    #[test]
    fn test_page_table_truncate() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2, 3, 4]);
        
        pt.truncate(3);
        
        assert_eq!(pt.len(), 3);
        assert_eq!(pt.get(3), None);
        assert_eq!(pt.get(4), None);
    }

    #[test]
    fn test_page_table_get_all_blocks() {
        let pt = PageTable::from_blocks(vec![0, 1, 2, 3]);
        let blocks = pt.get_all_blocks();
        
        assert_eq!(blocks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_page_table_manager() {
        let mut manager = PageTableManager::new();
        
        let id1 = manager.create_page_table(100);
        let id2 = manager.create_page_table(200);
        
        assert!(manager.contains(id1));
        assert!(manager.contains(id2));
        assert_eq!(manager.len(), 2);
        
        let pt = manager.get_page_table_mut(id1).unwrap();
        pt.append(0).unwrap();
        pt.append(1).unwrap();
        
        let pt = manager.get_page_table(id1).unwrap();
        assert_eq!(pt.len(), 2);
        
        manager.remove_page_table(id1);
        assert!(!manager.contains(id1));
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_page_table_fork() {
        let pt1 = PageTable::from_blocks(vec![0, 1, 2]);
        let pt2 = pt1.fork();
        
        assert_eq!(pt1.len(), pt2.len());
        assert_eq!(pt1.get_all_blocks(), pt2.get_all_blocks());
    }

    // 新增分支覆盖测试

    /// 测试 set 方法的正常分支
    #[test]
    fn test_page_table_set() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        
        pt.set(1, 99).unwrap();
        assert_eq!(pt.get(1), Some(99));
    }

    /// 测试 set 方法的越界错误分支
    #[test]
    fn test_page_table_set_out_of_range() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        
        let result = pt.set(5, 99);
        assert!(result.is_err(), "设置超出范围的槽位应返回错误");
    }

    /// 测试 invalidate 对不存在槽位的影响（不应 panic）
    #[test]
    fn test_page_table_invalidate_out_of_range() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        
        // 超出范围的 invalidate 不应 panic
        pt.invalidate(100);
        
        // 原有数据不应受影响
        assert_eq!(pt.len(), 3);
        assert_eq!(pt.valid_count(), 3);
    }

    /// 测试 pop 空页表返回 None
    #[test]
    fn test_page_table_pop_empty() {
        let mut pt = PageTable::new(10);
        
        assert_eq!(pt.pop(), None);
        assert_eq!(pt.len(), 0);
    }

    /// 测试 pop 无效条目返回 None
    #[test]
    fn test_page_table_pop_invalid_entry() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        pt.invalidate(2); // 使最后一个条目无效
        
        assert_eq!(pt.pop(), None); // 无效条目应返回 None
        assert_eq!(pt.len(), 2); // 但长度仍会减少
    }

    /// 测试 reserve 方法扩展容量
    #[test]
    fn test_page_table_reserve() {
        let mut pt = PageTable::new(5);
        assert_eq!(pt.capacity(), 5);
        
        pt.reserve(10);
        assert_eq!(pt.capacity(), 15); // 5 + 10
    }

    /// 测试 valid_count 方法计算有效条目数
    #[test]
    fn test_page_table_valid_count() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2, 3, 4]);
        assert_eq!(pt.valid_count(), 5);
        
        pt.invalidate(1);
        assert_eq!(pt.valid_count(), 4);
        
        pt.invalidate(3);
        assert_eq!(pt.valid_count(), 3);
    }

    /// 测试 is_valid 方法检查槽位是否有效
    #[test]
    fn test_page_table_is_valid() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        
        assert!(pt.is_valid(0));
        assert!(pt.is_valid(1));
        assert!(pt.is_valid(2));
        
        pt.invalidate(1);
        
        assert!(pt.is_valid(0));
        assert!(!pt.is_valid(1));
        assert!(pt.is_valid(2));
        
        // 越界访问应返回 false
        assert!(!pt.is_valid(100));
    }

    /// 测试 truncate 到更大长度（不执行任何操作）
    #[test]
    fn test_page_table_truncate_larger() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        let len_before = pt.len();
        
        pt.truncate(10); // 大于当前长度
        
        assert_eq!(pt.len(), len_before, "截断到更大长度不应改变");
    }

    /// 测试 clear 方法清空页表
    #[test]
    fn test_page_table_clear() {
        let mut pt = PageTable::from_blocks(vec![0, 1, 2, 3, 4]);
        
        pt.clear();
        
        assert_eq!(pt.len(), 0);
        assert!(pt.is_empty());
    }

    /// 测试 last_block 方法获取最后一个有效块
    #[test]
    fn test_page_table_last_block() {
        let pt = PageTable::from_blocks(vec![0, 1, 2]);
        assert_eq!(pt.last_block(), Some(2));
        
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        pt.invalidate(2);
        assert_eq!(pt.last_block(), Some(1)); // 最后一个有效块是 1
        
        let mut pt = PageTable::from_blocks(vec![0, 1, 2]);
        pt.invalidate(0);
        pt.invalidate(1);
        pt.invalidate(2);
        assert_eq!(pt.last_block(), None); // 没有有效块
    }

    /// 测试 append_blocks 批量添加
    #[test]
    fn test_page_table_append_blocks() {
        let mut pt = PageTable::new(10);
        
        pt.append(0).unwrap();
        pt.append_blocks(vec![1, 2, 3]).unwrap();
        
        assert_eq!(pt.len(), 4);
        assert_eq!(pt.get(0), Some(0));
        assert_eq!(pt.get(3), Some(3));
    }

    /// 测试 append_blocks 溢出错误
    #[test]
    fn test_page_table_append_blocks_overflow() {
        let mut pt = PageTable::new(5);
        
        pt.append_blocks(vec![0, 1, 2, 3, 4]).unwrap(); // 填满
        assert!(pt.append_blocks(vec![5]).is_err()); // 溢出
    }

    /// 测试 Default trait 实现
    #[test]
    fn test_page_table_default() {
        let pt = PageTable::default();
        assert_eq!(pt.capacity(), 1024);
        assert!(pt.is_empty());
    }

    /// 测试 PageTableEntry 的 new 和 invalid 方法
    #[test]
    fn test_page_table_entry() {
        let valid_entry = PageTableEntry::new(42);
        assert!(valid_entry.valid);
        assert_eq!(valid_entry.block_id, 42);
        
        let invalid_entry = PageTableEntry::invalid();
        assert!(!invalid_entry.valid);
        assert_eq!(invalid_entry.block_id, 0);
    }

    /// 测试 PageTableManager 的 request_ids、is_empty、clear 方法
    #[test]
    fn test_page_table_manager_utilities() {
        let mut manager = PageTableManager::new();
        
        // 初始状态
        assert!(manager.is_empty());
        assert!(manager.request_ids().is_empty());
        
        // 创建页表
        let id1 = manager.create_page_table(100);
        let id2 = manager.create_page_table(200);
        
        assert!(!manager.is_empty());
        let ids = manager.request_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
        
        // 清空
        manager.clear();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
    }
}
