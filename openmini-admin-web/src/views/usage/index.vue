<script setup lang="ts">
import { ref, computed } from 'vue'
import { DataLine, Timer, WarningFilled } from '@element-plus/icons-vue'

const timeRange = ref('7d')

const statsData = ref([
  { label: '总 Token 数', value: '12.8M', icon: DataLine, color: '#409EFF' },
  { label: '总请求数', value: '45,230', icon: Timer, color: '#67C23A' },
  { label: '平均延迟', value: '123ms', icon: WarningFilled, color: '#E6A23C' },
  { label: '成本估算', value: '$286.50', icon: '', color: '#F56C6C' }
])

const chartGranularity = ref('day')

const tokenTrendData = computed(() => ({
  xAxis: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
  series: [
    {
      name: 'Prompt Tokens',
      data: [120000, 145000, 132000, 158000, 142000, 98000, 85000],
      color: '#409EFF'
    },
    {
      name: 'Completion Tokens',
      data: [380000, 420000, 395000, 450000, 410000, 280000, 245000],
      color: '#67C23A'
    }
  ]
}))

const modelDistribution = ref([
  { name: 'Qwen-14B', value: 45 },
  { name: 'Llama-3-8B', value: 28 },
  { name: 'Baichuan2-7B', value: 18 },
  { name: 'Yi-34B', value: 9 }
])

const apiDistribution = ref([
  { name: 'Chat Completions', value: 65 },
  { name: 'Embeddings', value: 20 },
  { name: 'Models', value: 10 },
  { name: 'Others', value: 5 }
])

const topUsers = ref([
  { rank: 1, name: 'user_dev_001', requests: 12560, tokens: '2.3M', lastActive: '2026-04-09 14:30' },
  { rank: 2, name: 'user_prod_002', requests: 10234, tokens: '1.8M', lastActive: '2026-04-09 13:45' },
  { rank: 3, name: 'user_test_003', requests: 8670, tokens: '1.5M', lastActive: '2026-04-09 12:20' },
  { rank: 4, name: 'user_api_004', requests: 6540, tokens: '1.1M', lastActive: '2026-04-09 11:55' },
  { rank: 5, name: 'user_demo_005', requests: 4320, tokens: '780K', lastActive: '2026-04-09 10:30' }
])

const topApiKeys = ref([
  { rank: 1, name: 'sk-prod-key-001', requests: 18920, tokens: '3.5M', lastActive: '2026-04-09 14:28' },
  { rank: 2, name: 'sk-dev-key-002', requests: 14350, tokens: '2.6M', lastActive: '2026-04-09 14:15' },
  { rank: 3, name: 'sk-test-key-003', requests: 9870, tokens: '1.7M', lastActive: '2026-04-09 13:50' }
])

const costTable = ref([
  { month: '2026-01', prompt: '4.2M', completion: '12.6M', total: '16.8M', unitPrice: '$0.02/M', cost: '$38.40' },
  { month: '2026-02', prompt: '5.1M', completion: '14.8M', total: '19.9M', unitPrice: '$0.02/M', cost: '$45.36' },
  { month: '2026-03', prompt: '6.3M', completion: '17.2M', total: '23.5M', unitPrice: '$0.02/M', cost: '$53.76' },
  { month: '2026-04', prompt: '2.8M', completion: '8.4M', total: '11.2M', unitPrice: '$0.02/M', cost: '$25.60' }
])

const maxRequests = computed(() => Math.max(...topUsers.value.map(u => u.requests)))
</script>

<template>
  <div class="usage-container">
    <PageHeader title="用量分析" subtitle="API调用统计与成本分析">
      <template #extra>
        <el-radio-group v-model="timeRange" size="small">
          <el-radio-button value="7d">近7天</el-radio-button>
          <el-radio-button value="30d">近30天</el-radio-button>
          <el-radio-button value="90d">近90天</el-radio-button>
        </el-radio-group>
      </template>
    </PageHeader>

    <!-- 统计概览卡片 -->
    <el-row :gutter="20" class="stats-row">
      <el-col v-for="(stat, index) in statsData" :key="index" :xs="24" :sm="12" :md="6">
        <StatCard :title="stat.label" :value="stat.value" :color="'default'" :icon="stat.icon" />
      </el-col>
    </el-row>

    <!-- Token趋势图 -->
    <el-card shadow="hover" class="trend-card">
      <template #header>
        <div class="card-header-flex">
          <span>Token 用量趋势</span>
          <el-radio-group v-model="chartGranularity" size="small">
            <el-radio-button value="day">按日</el-radio-button>
            <el-radio-button value="week">按周</el-radio-button>
            <el-radio-button value="month">按月</el-radio-button>
          </el-radio-group>
        </div>
      </template>
      <LineChart :data="tokenTrendData" :height="320" :area-style="true" />
    </el-card>

    <!-- 调用分布饼图 -->
    <el-row :gutter="20" class="distribution-row">
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header><span>按模型分布</span></template>
          <PieChart :data="modelDistribution" :height="280" />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header><span>按接口类型分布</span></template>
          <PieChart :data="apiDistribution" :height="280" />
        </el-card>
      </el-col>
    </el-row>

    <!-- Top排行榜 -->
    <el-card shadow="hover" class="ranking-card">
      <template #header><span>Top 排行榜</span></template>
      <el-tabs>
        <el-tab-pane label="用户排行" name="users">
          <el-table :data="topUsers" stripe>
            <el-table-column prop="rank" label="#" width="50" />
            <el-table-column prop="name" label="名称" min-width="180">
              <template #default="{ row }">
                <code>{{ row.name }}</code>
              </template>
            </el-table-column>
            <el-table-column prop="requests" label="请求数量" width="180">
              <template #default="{ row }">
                <div class="progress-cell">
                  <el-progress
                    :percentage="(row.requests / maxRequests) * 100"
                    :stroke-width="8"
                    :show-text="false"
                    style="flex: 1"
                  />
                  <span class="progress-value">{{ row.requests.toLocaleString() }}</span>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="tokens" label="Token数" width="100" />
            <el-table-column prop="lastActive" label="最后活跃" width="150" />
          </el-table>
        </el-tab-pane>
        <el-tab-pane label="API Key排行" name="keys">
          <el-table :data="topApiKeys" stripe>
            <el-table-column prop="rank" label="#" width="50" />
            <el-table-column prop="name" label="Key名称" min-width="200">
              <template #default="{ row }">
                <code>{{ row.name.slice(0, 12) }}...</code>
              </template>
            </el-table-column>
            <el-table-column prop="requests" label="请求数量" width="180">
              <template #default="{ row }">
                <div class="progress-cell">
                  <el-progress
                    :percentage="(row.requests / 20000) * 100"
                    :stroke-width="8"
                    :show-text="false"
                    style="flex: 1"
                  />
                  <span class="progress-value">{{ row.requests.toLocaleString() }}</span>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="tokens" label="Token数" width="100" />
            <el-table-column prop="lastActive" label="最后活跃" width="150" />
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 成本报表 -->
    <el-card shadow="hover" class="cost-card">
      <template #header><span>月度成本报表</span></template>
      <el-table :data="costTable" stripe>
        <el-table-column prop="month" label="月份" width="110" />
        <el-table-column prop="prompt" label="Prompt Tokens" />
        <el-table-column prop="completion" label="Completion Tokens" />
        <el-table-column prop="total" label="Total Tokens" />
        <el-table-column prop="unitPrice" label="单价" width="100" />
        <el-table-column prop="cost" label="成本" width="100">
          <template #default="{ row }">
            <strong>{{ row.cost }}</strong>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.usage-container {
  .stats-row {
    margin-bottom: $spacing-lg;
  }

  .trend-card,
  .distribution-row,
  .ranking-card,
  .cost-card {
    margin-bottom: $spacing-lg;
  }

  .card-header-flex {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .progress-cell {
    display: flex;
    align-items: center;
    gap: $spacing-sm;

    .progress-value {
      font-weight: bold;
      min-width: 70px;
      text-align: right;
    }
  }

  code {
    background: #f0f2f5;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: $font-mono;
    font-size: 12px;
  }
}
</style>
