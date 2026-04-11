//! ESS 缓存淘汰策略模块

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum EvictionPolicy {
    LRU,      // Least Recently Used
    LFU,      // Least Frequently Used
    Adaptive, // 自适应策略
}

pub struct LruEviction {
    queue: VecDeque<String>,
    access_time: HashMap<String, u64>,
}

impl LruEviction {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            access_time: HashMap::new(),
        }
    }

    pub fn access(&mut self, key: &str) {
        if let Some(pos) = self.queue.iter().position(|k| k == key) {
            self.queue.remove(pos);
        }
        self.queue.push_back(key.to_string());
        self.access_time
            .insert(key.to_string(), self.queue.len() as u64);
    }

    pub fn evict(&mut self) -> Option<String> {
        self.queue.pop_front()
    }
}

impl Default for LruEviction {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LfuEviction {
    frequency: HashMap<String, u32>,
    access_order: Vec<String>,
}

impl LfuEviction {
    pub fn new() -> Self {
        Self {
            frequency: HashMap::new(),
            access_order: Vec::new(),
        }
    }

    pub fn access(&mut self, key: &str) {
        let count = self.frequency.entry(key.to_string()).or_insert(0);
        *count += 1;

        if !self.access_order.contains(&key.to_string()) {
            self.access_order.push(key.to_string());
        }
    }

    pub fn evict(&mut self) -> Option<String> {
        if self.access_order.is_empty() {
            return None;
        }

        let min_key = self
            .access_order
            .iter()
            .min_by_key(|k| self.frequency.get(*k).unwrap_or(&0))
            .cloned();

        if let Some(key) = min_key.clone() {
            self.access_order.retain(|k| k != &key);
            self.frequency.remove(&key);
        }

        min_key
    }
}

impl Default for LfuEviction {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AdaptiveEviction {
    lru: LruEviction,
    lfu: LfuEviction,
    mode: EvictionPolicy,
    hit_rate_lru: f64,
    hit_rate_lfu: f64,
}

impl AdaptiveEviction {
    pub fn new() -> Self {
        Self {
            lru: LruEviction::new(),
            lfu: LfuEviction::new(),
            mode: EvictionPolicy::LRU,
            hit_rate_lru: 0.0,
            hit_rate_lfu: 0.0,
        }
    }

    pub fn access(&mut self, key: &str) {
        self.lru.access(key);
        self.lfu.access(key);
    }

    pub fn record_hit(&mut self, policy: EvictionPolicy) {
        match policy {
            EvictionPolicy::LRU => self.hit_rate_lru += 1.0,
            EvictionPolicy::LFU => self.hit_rate_lfu += 1.0,
            EvictionPolicy::Adaptive => {
                unreachable!("policy should never be Adaptive in record_hit")
            }
        }

        if self.hit_rate_lru + self.hit_rate_lfu > 100.0 {
            if self.hit_rate_lfu > self.hit_rate_lru * 1.2 {
                self.mode = EvictionPolicy::LFU;
            } else if self.hit_rate_lru > self.hit_rate_lfu * 1.2 {
                self.mode = EvictionPolicy::LRU;
            }
            self.hit_rate_lru = 0.0;
            self.hit_rate_lfu = 0.0;
        }
    }

    pub fn evict(&mut self) -> Option<String> {
        match self.mode {
            EvictionPolicy::LRU => self.lru.evict(),
            EvictionPolicy::LFU => self.lfu.evict(),
            EvictionPolicy::Adaptive => unreachable!("mode should never be Adaptive"),
        }
    }
}

impl Default for AdaptiveEviction {
    fn default() -> Self {
        Self::new()
    }
}
