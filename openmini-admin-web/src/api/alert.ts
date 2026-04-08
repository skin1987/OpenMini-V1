import { get, post, put, del } from '@/utils/request'
import type {
  AlertRule,
  CreateAlertRuleRequest,
  UpdateAlertRuleRequest,
  AlertRecord,
  AlertQueryParams,
  AcknowledgeAlertRequest,
  ResolveAlertRequest,
  ToggleAlertRuleResponse,
  TestAlertRequest,
  TestAlertResponse,
  AlertSummaryData
} from '@/types/api/alert'

/**
 * 获取告警规则列表
 * @returns 所有告警规则（包含触发次数等统计）
 */
export function getAlertRules(): Promise<AlertRule[]> {
  return get<AlertRule[]>('/admin/alerts/rules')
}

/**
 * 创建告警规则
 * @param data - 规则配置（名称、条件、通知渠道等）
 * @returns 创建成功的规则信息
 */
export function createAlertRule(data: CreateAlertRuleRequest): Promise<AlertRule> {
  return post<AlertRule>('/admin/alerts/rules', data)
}

/**
 * 更新告警规则
 * @param id - 规则 ID
 * @param data - 需要更新的字段
 * @returns 更新后的规则信息
 */
export function updateAlertRule(id: string, data: UpdateAlertRuleRequest): Promise<AlertRule> {
  return put<AlertRule>(`/admin/alerts/rules/${id}`, data)
}

/**
 * 删除告警规则
 * @param id - 规则 ID
 * @returns 删除结果
 */
export function deleteAlertRule(id: string): Promise<{ success: boolean; message: string }> {
  return del(`/admin/alerts/rules/${id}`)
}

/**
 * 启用/禁用告警规则
 * @param id - 规则 ID
 * @param enabled - 是否启用
 * @returns 操作结果和新状态
 */
export function toggleAlertRule(id: string, enabled: boolean): Promise<ToggleAlertRuleResponse> {
  return post<ToggleAlertRuleResponse>(`/admin/alerts/rules/${id}/toggle`, { enabled })
}

/**
 * 获取告警记录列表
 * @param params - 分页和筛选条件
 * @returns 分页的告警记录列表
 */
export function getAlertRecords(params?: AlertQueryParams): Promise<{
  items: AlertRecord[]
  total: number
  page: number
  page_size: number
}> {
  return get('/admin/alerts/records', { params })
}

/**
 * 确认告警（标记为已读）
 * @param id - 告警记录 ID
 * @param data - 确认人信息和备注
 * @returns 确认结果
 */
export function acknowledgeAlert(id: string, data: AcknowledgeAlertRequest): Promise<{ success: boolean; message: string }> {
  return post(`/admin/alerts/records/${id}/ack`, data)
}

/**
 * 解决告警（标记为已解决）
 * @param id - 告警记录 ID
 * @param data - 解决人和解决方案说明
 * @returns 解决结果
 */
export function resolveAlert(id: string, data: ResolveAlertRequest): Promise<{ success: boolean; message: string }> {
  return post(`/admin/alerts/records/${id}/resolve`, data)
}

/**
 * 获取告警统计摘要
 * @returns 各级别告警数量、趋势分析等汇总信息
 */
export function getAlertSummary(): Promise<AlertSummaryData> {
  return get<AlertSummaryData>('/admin/alerts/summary')
}

/**
 * 测试告警通知
 * @param data - 测试配置（目标渠道、测试消息）
 * @returns 测试发送结果和投递详情
 */
export function testAlert(data: TestAlertRequest): Promise<TestAlertResponse> {
  return post<TestAlertResponse>('/admin/alerts/test', data)
}
