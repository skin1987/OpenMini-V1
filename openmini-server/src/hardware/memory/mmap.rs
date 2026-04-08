//! 内存映射文件 - 高效的文件访问
//!
//! 提供基于 mmap 的文件访问，避免数据拷贝，适合大文件读取。
//! 主要用于加载模型权重文件。
//!
//! # 示例
//! ```
//! use std::path::Path;
//!
//! // 从路径打开
//! let mmap = MmapFile::open(Path::new("model.bin"))?;
//!
//! // 从已打开的文件创建
//! let file = File::open("model.bin")?;
//! let mmap = MmapFile::from_file(file)?;
//! ```

#![allow(dead_code)]

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;

/// 内存映射文件
///
/// 将文件内容映射到内存地址空间，支持零拷贝访问。
///
/// # 特点
/// - 零拷贝访问文件内容
/// - 自动处理文件关闭
/// - 支持从路径或已打开的文件创建
///
/// # 线程安全
/// `as_slice()` 返回的切片是只读的，可安全跨线程共享。
pub struct MmapFile {
    /// 内存映射区域
    mmap: memmap2::Mmap,
}

impl MmapFile {
    /// 打开并映射文件
    ///
    /// # 参数
    /// - `path`: 文件路径
    ///
    /// # 返回
    /// 成功返回 MmapFile 实例，失败返回 IO 错误
    ///
    /// # 示例
    /// ```
    /// use std::path::Path;
    /// let mmap = MmapFile::open(Path::new("model.bin"))?;
    /// ```
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        Self::from_file(file)
    }

    /// 从已打开的文件创建内存映射
    ///
    /// # 参数
    /// - `file`: 已打开的文件（必须具有读取权限）
    ///
    /// # 返回
    /// 成功返回 MmapFile 实例，失败返回 IO 错误
    ///
    /// # 示例
    /// ```
    /// let file = File::open("model.bin")?;
    /// let mmap = MmapFile::from_file(file)?;
    /// ```
    pub fn from_file(file: File) -> io::Result<Self> {
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    /// 获取文件内容的字节切片
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// 获取文件大小
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// 检查文件是否为空
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// 获取指定范围的数据
    ///
    /// # 参数
    /// - `offset`: 起始偏移
    /// - `len`: 数据长度
    ///
    /// # 返回
    /// 成功返回数据切片，越界返回 None
    pub fn get_range(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if offset.saturating_add(len) <= self.mmap.len() {
            Some(&self.mmap[offset..offset + len])
        } else {
            None
        }
    }
}

impl Deref for MmapFile {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.mmap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// 测试：从路径成功打开并映射文件（正常分支）
    #[test]
    fn test_open_from_path() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello, mmap!").unwrap();

        let mmap = MmapFile::open(file.path()).expect("应该成功打开文件");
        assert_eq!(mmap.len(), 12);
        assert!(!mmap.is_empty());
    }

    /// 测试：从已打开的文件对象创建mmap（from_file分支）
    #[test]
    fn test_from_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test data").unwrap();
        let file_handle = file.reopen().unwrap();

        let mmap = MmapFile::from_file(file_handle).expect("应该从文件对象创建成功");
        assert_eq!(mmap.len(), 9);
    }

    /// 测试：空文件的mmap（边界条件：零长度）
    #[test]
    fn test_empty_file() {
        let file = NamedTempFile::new().unwrap();

        let mmap = MmapFile::open(file.path()).expect("空文件也应该能打开");
        assert_eq!(mmap.len(), 0);
        assert!(mmap.is_empty());
        assert_eq!(mmap.as_slice().len(), 0);
    }

    /// 测试：as_slice返回正确的字节内容
    #[test]
    fn test_as_slice_content() {
        let mut file = NamedTempFile::new().unwrap();
        let data = b"Test content for slice";
        file.write_all(data).unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        assert_eq!(mmap.as_slice(), *data);
    }

    /// 测试：get_range正常范围访问（核心功能分支）
    #[test]
    fn test_get_range_normal() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"0123456789").unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        let range = mmap.get_range(2, 5).expect("应该获取到范围");
        assert_eq!(range, b"23456");
    }

    /// 测试：get_range从开头获取（offset=0边界）
    #[test]
    fn test_get_range_from_start() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"abcdef").unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        let range = mmap.get_range(0, 3).expect("应该能从头开始");
        assert_eq!(range, b"abc");
    }

    /// 测试：get_range获取到末尾（len=0边界）
    #[test]
    fn test_get_range_zero_length() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"hello").unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        let range = mmap.get_range(3, 0).expect("零长度范围应该有效");
        assert_eq!(range.len(), 0);
    }

    /// 测试：get_range越界返回None（错误路径）
    #[test]
    fn test_get_range_out_of_bounds() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"short").unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();

        // offset超出范围
        assert!(mmap.get_range(100, 1).is_none());
        // len超出范围
        assert!(mmap.get_range(2, 100).is_none());
        // offset + len刚好等于长度（合法）
        assert!(mmap.get_range(5, 0).is_some());
        // offset + len刚好超过1（非法）
        assert!(mmap.get_range(5, 1).is_none());
    }

    /// 测试：Deref trait自动解引用为[u8]切片
    #[test]
    fn test_deref_trait() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"deref test").unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        // 通过Deref自动转换为&[u8]
        let slice: &[u8] = &*mmap;
        assert_eq!(slice, b"deref test");
        assert_eq!(slice.len(), 10);
    }

    /// 测试：文件不存在时返回IO错误（错误处理分支）
    #[test]
    fn test_open_nonexistent_file() {
        let result = MmapFile::open(Path::new("/nonexistent/path/file.bin"));
        assert!(result.is_err(), "不存在的文件应该返回错误");
    }

    /// 测试：较大文件的mmap操作（性能和稳定性验证，<20KB）
    #[test]
    fn test_larger_file_mmap() {
        let mut file = NamedTempFile::new().unwrap();
        // 写入约16KB数据（远小于20KB限制）
        let data: Vec<u8> = (0..255).cycle().take(16384).collect();
        file.write_all(&data).unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        assert_eq!(mmap.len(), 16384);

        // 验证不同位置的数据正确性
        assert_eq!(mmap.get_range(0, 256).unwrap(), &data[..256]);
        assert_eq!(mmap.get_range(8000, 1000).unwrap(), &data[8000..9000]);
        assert_eq!(mmap.get_range(16000, 384).unwrap(), &data[16000..]);
    }

    // ==================== 新增分支覆盖率测试 ====================

    /// 测试：get_range - 读取整个文件内容（offset=0, len=文件长度）
    #[test]
    fn test_get_range_full_file() {
        let mut file = NamedTempFile::new().unwrap();
        let data = b"Complete file content";
        file.write_all(data).unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        let full_content = mmap.get_range(0, data.len()).expect("应该能读取完整文件");
        assert_eq!(full_content, *data);
    }

    /// 测试：get_range - 读取最后一个字节（offset=len-1, len=1 边界）
    #[test]
    fn test_get_range_last_byte() {
        let mut file = NamedTempFile::new().unwrap();
        let data = b"ABCDE";
        file.write_all(data).unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        let last_byte = mmap
            .get_range(data.len() - 1, 1)
            .expect("应该能读取最后一个字节");
        assert_eq!(last_byte, b"E");
    }

    /// 测试：from_file - 空文件的mmap创建（from_file + 空文件组合分支）
    #[test]
    fn test_from_file_empty() {
        let file = NamedTempFile::new().unwrap();
        let file_handle = file.reopen().unwrap();

        let mmap = MmapFile::from_file(file_handle).expect("空文件也应该能从file对象创建");
        assert_eq!(mmap.len(), 0);
        assert!(mmap.is_empty());
    }

    /// 测试：Deref trait - 空文件解引用为空切片（Deref + 空文件组合）
    #[test]
    fn test_deref_empty_file() {
        let file = NamedTempFile::new().unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        let slice: &[u8] = &*mmap; // 通过Deref自动转换

        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());
    }

    /// 测试：as_slice - 空文件返回空切片（as_slice + 空文件组合）
    #[test]
    fn test_as_slice_empty_file() {
        let file = NamedTempFile::new().unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        let slice = mmap.as_slice();

        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());
    }

    /// 测试：二进制数据文件（非ASCII字符，特殊数据类型分支）
    #[test]
    fn test_binary_data_file() {
        let mut file = NamedTempFile::new().unwrap();
        // 包含所有可能的字节值（0-255）
        let binary_data: Vec<u8> = (0..=255).collect();
        file.write_all(&binary_data).unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        assert_eq!(mmap.len(), 256); // 0-255 共256个字节

        // 验证二进制数据完整性
        assert_eq!(mmap.as_slice(), binary_data.as_slice());

        // 验证可以正确访问每个字节
        for (i, &byte) in mmap.as_slice().iter().enumerate() {
            assert_eq!(byte as usize, i, "Byte at index {} should equal {}", i, i);
        }
    }

    /// 测试：get_range - 双零边界（offset=0, len=0）
    #[test]
    fn test_get_range_double_zero() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"data").unwrap();

        let mmap = MmapFile::open(file.path()).unwrap();
        let range = mmap.get_range(0, 0).expect("offset=0, len=0 应该有效");
        assert_eq!(range.len(), 0);
    }

    /// 测试：多次打开同一文件的独立性（多次调用分支）
    #[test]
    fn test_multiple_opens_independence() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"shared content").unwrap();

        // 多次打开同一文件，应该是独立的MmapFile实例
        let mmap1 = MmapFile::open(file.path()).unwrap();
        let mmap2 = MmapFile::open(file.path()).unwrap();
        let mmap3 = MmapFile::from_file(file.reopen().unwrap()).unwrap();

        // 所有实例都应该能独立访问相同的内容
        assert_eq!(mmap1.len(), mmap2.len());
        assert_eq!(mmap2.len(), mmap3.len());
        assert_eq!(mmap1.as_slice(), mmap2.as_slice());
        assert_eq!(mmap2.as_slice(), mmap3.as_slice());
    }

    /// 测试：文件路径包含Unicode/中文（国际化支持分支）
    #[test]
    fn test_unicode_filename() {
        use std::path::PathBuf;

        // 在临时目录中创建带中文的文件名
        let temp_dir = tempfile::tempdir().unwrap();
        let unicode_path = PathBuf::from(temp_dir.path()).join("测试文件.bin");

        // 写入并读取
        {
            use std::fs::File;
            use std::io::Write;
            let mut f = File::create(&unicode_path).unwrap();
            f.write_all(b"unicode test").unwrap();
        }

        let result = MmapFile::open(&unicode_path);
        assert!(result.is_ok(), "应该能打开Unicode文件名的文件");

        let mmap = result.unwrap();
        assert_eq!(mmap.as_slice(), b"unicode test");
    }
}
