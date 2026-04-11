<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UserFilled, Edit, Delete, Lock, Plus } from '@element-plus/icons-vue'
import { useUserStore } from '@/stores/user'
import {
  getUserList,
  createUser,
  updateUser,
  deleteUser,
  updateUserRole,
  updateUserStatus,
  resetUserPassword
} from '@/api/user'
import type {
  UserInfo,
  UserQueryParams,
  CreateUserRequest,
  UpdateUserRequest
} from '@/types/api/user'
import type { UserRole } from '@/types'

const userStore = useUserStore()
const loading = ref(false)
const userList = ref<UserInfo[]>([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)
const keyword = ref('')
const roleFilter = ref('')
const statusFilter = ref('')

const dialogVisible = ref(false)
const dialogTitle = ref('创建用户')
const isEdit = ref(false)
const currentEditUser = ref<UserInfo | null>(null)

const form = reactive({
  username: '',
  email: '',
  password: '',
  role: 'viewer' as UserRole,
  status: true
})

const queryParams = computed<UserQueryParams>(() => ({
  page: currentPage.value,
  page_size: pageSize.value,
  keyword: keyword.value || undefined,
  role: (roleFilter.value as UserRole) || undefined,
  status: (statusFilter.value as any) || undefined
}))

const isAdmin = computed(() => userStore.role === 'admin')

onMounted(() => {
  fetchUserList()
})

async function fetchUserList() {
  loading.value = true
  try {
    const res = await getUserList(queryParams.value)
    userList.value = res.items
    total.value = res.total
  } catch (error) {
    ElMessage.error('获取用户列表失败')
  } finally {
    loading.value = false
  }
}

function handleSearch() {
  currentPage.value = 1
  fetchUserList()
}

function handleFilterChange() {
  currentPage.value = 1
  fetchUserList()
}

function handleSizeChange(val: number) {
  pageSize.value = val
  currentPage.value = 1
  fetchUserList()
}

function handleCurrentChange(val: number) {
  currentPage.value = val
  fetchUserList()
}

function openCreateDialog() {
  isEdit.value = false
  dialogTitle.value = '创建用户'
  form.username = ''
  form.email = ''
  form.password = ''
  form.role = 'viewer'
  form.status = true
  currentEditUser.value = null
  dialogVisible.value = true
}

function openEditDialog(row: UserInfo) {
  isEdit.value = true
  dialogTitle.value = '编辑用户'
  currentEditUser.value = row
  form.username = row.username
  form.email = row.email
  form.password = ''
  form.role = row.role
  form.status = row.status === 'active'
  dialogVisible.value = true
}

async function handleSubmit() {
  if (!form.username.trim()) {
    ElMessage.warning('请输入用户名')
    return
  }
  if (!form.email.trim()) {
    ElMessage.warning('请输入邮箱')
    return
  }
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRegex.test(form.email)) {
    ElMessage.warning('邮箱格式不正确')
    return
  }
  if (!isEdit.value && !form.password) {
    ElMessage.warning('请输入密码')
    return
  }
  if (form.password && (form.password.length < 6 || form.password.length > 30)) {
    ElMessage.warning('密码长度应在6-30个字符之间')
    return
  }
  if (form.username.length < 3 || form.username.length > 20) {
    ElMessage.warning('用户名长度应在3-20个字符之间')
    return
  }

  try {
    if (isEdit.value && currentEditUser.value) {
      const data: UpdateUserRequest = {
        email: form.email,
        role: form.role
      }
      await updateUser(currentEditUser.value.id, data)

      if (form.status !== (currentEditUser.value.status === 'active')) {
        await updateUserStatus(currentEditUser.value.id, form.status ? 'active' : 'disabled')
      }

      ElMessage.success('更新成功')
    } else {
      const data: CreateUserRequest = {
        username: form.username.trim(),
        email: form.email.trim(),
        password: form.password,
        display_name: form.username,
        role: form.role
      }
      await createUser(data)
      ElMessage.success('创建成功')
    }

    dialogVisible.value = false
    fetchUserList()
  } catch (error) {
    ElMessage.error(isEdit.value ? '更新失败' : '创建失败')
  }
}

async function handleRoleChange(row: UserInfo, newRole: UserRole) {
  try {
    await updateUserRole(row.id, newRole)
    ElMessage.success(`已将 ${row.username} 的角色修改为 ${newRole}`)
    fetchUserList()
  } catch (error) {
    ElMessage.error('修改角色失败')
  }
}

async function handleToggleStatus(row: UserInfo) {
  const newStatus = row.status === 'active' ? 'disabled' : 'active'
  try {
    await updateUserStatus(row.id, newStatus)
    ElMessage.success(`已${newStatus === 'active' ? '启用' : '禁用'}用户 ${row.username}`)
    fetchUserList()
  } catch (error) {
    ElMessage.error('操作失败')
  }
}

async function handleResetPassword(row: UserInfo) {
  try {
    await ElMessageBox.confirm(
      `确定要重置用户 ${row.username} 的密码吗？`,
      '重置密码确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    const result = await resetUserPassword(row.id)
    ElMessageBox.alert(
      `临时密码: ${result.temporary_password || '已生成'}\n有效期至: ${result.expires_at || '请查看邮件'}`,
      '重置成功',
      {
        confirmButtonText: '我知道了',
        type: 'success'
      }
    )
    ElMessage.success('密码已重置')
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('重置密码失败')
    }
  }
}

async function handleDelete(row: UserInfo) {
  try {
    await ElMessageBox.confirm(
      `确定要删除用户 ${row.username} 吗？此操作不可恢复！`,
      '删除确认',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'error',
        confirmButtonClass: 'el-button--danger'
      }
    )
    await deleteUser(row.id)
    ElMessage.success('删除成功')
    fetchUserList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

function getRoleType(role: string): 'danger' | 'warning' | 'info' {
  const map: Record<string, 'danger' | 'warning' | 'info'> = {
    admin: 'danger',
    operator: 'warning',
    viewer: 'info'
  }
  return map[role] || 'info'
}

function getRoleText(role: string): string {
  const map: Record<string, string> = {
    admin: '管理员',
    operator: '运维人员',
    viewer: '访客'
  }
  return map[role] || role
}

function formatTime(time?: string) {
  if (!time) return '-'
  return time.slice(0, 16).replace('T', ' ')
}
</script>

<template>
  <div class="page-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <h2>用户管理</h2>
    </div>

    <!-- 工具栏 -->
    <el-card v-if="isAdmin" class="toolbar-card" shadow="never">
      <div class="toolbar">
        <el-input
          v-model="keyword"
          placeholder="搜索用户名或邮箱"
          clearable
          style="width: 240px"
          @keyup.enter="handleSearch"
          @clear="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>

        <el-select
          v-model="roleFilter"
          placeholder="角色筛选"
          clearable
          style="width: 140px"
          @change="handleFilterChange"
        >
          <el-option label="全部" value="" />
          <el-option label="管理员" value="admin" />
          <el-option label="运维人员" value="operator" />
          <el-option label="访客" value="viewer" />
        </el-select>

        <el-select
          v-model="statusFilter"
          placeholder="状态筛选"
          clearable
          style="width: 140px"
          @change="handleFilterChange"
        >
          <el-option label="全部" value="" />
          <el-option label="启用" value="active" />
          <el-option label="禁用" value="disabled" />
        </el-select>

        <el-button type="primary" @click="openCreateDialog" v-permission="['admin']">
          <el-icon><Plus /></el-icon>
          创建用户
        </el-button>
      </div>
    </el-card>

    <!-- 数据表格 -->
    <el-card shadow="never" class="table-card">
      <el-table
        :data="userList"
        v-loading="loading"
        stripe
        border
        style="width: 100%"
      >
        <el-table-column label="头像" width="80" align="center">
          <template #default>
            <el-avatar :size="36" :icon="UserFilled" />
          </template>
        </el-table-column>

        <el-table-column prop="username" label="用户名" min-width="120" />

        <el-table-column prop="email" label="邮箱" min-width="180" />

        <el-table-column prop="role" label="角色" width="110">
          <template #default="{ row }">
            <el-tag :type="getRoleType(row.role)" effect="dark">
              {{ getRoleText(row.role) }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="状态" width="90" align="center">
          <template #default="{ row }">
            <el-tag :type="row.status === 'active' ? 'success' : 'danger'" effect="dark">
              {{ row.status === 'active' ? '启用' : '禁用' }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="最后登录" width="160">
          <template #default="{ row }">
            {{ formatTime(row.last_login_at) }}
          </template>
        </el-table-column>

        <el-table-column label="创建时间" width="160">
          <template #default="{ row }">
            {{ formatTime(row.created_at) }}
          </template>
        </el-table-column>

        <el-table-column
          v-if="isAdmin"
          label="操作"
          width="280"
          fixed="right"
        >
          <template #default="{ row }">
            <el-tooltip content="编辑" placement="top">
              <el-button link type="primary" size="small" @click="openEditDialog(row)">
                <el-icon><Edit /></el-icon>
              </el-button>
            </el-tooltip>

            <el-dropdown trigger="click" @command="(cmd: UserRole) => handleRoleChange(row, cmd)">
              <el-button link type="warning" size="small">
                角色<el-icon class="el-icon--right"><ArrowDown /></el-icon>
              </el-button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="admin" :disabled="!isAdmin">管理员</el-dropdown-item>
                  <el-dropdown-item command="operator">运维人员</el-dropdown-item>
                  <el-dropdown-item command="viewer">访客</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>

            <el-tooltip :content="row.status === 'active' ? '禁用' : '启用'" placement="top">
              <el-button
                link
                :type="row.status === 'active' ? 'warning' : 'success'"
                size="small"
                @click="handleToggleStatus(row)"
              >
                {{ row.status === 'active' ? '禁用' : '启用' }}
              </el-button>
            </el-tooltip>

            <el-tooltip content="重置密码" placement="top">
              <el-button link type="info" size="small" @click="handleResetPassword(row)">
                <el-icon><Lock /></el-icon>
              </el-button>
            </el-tooltip>

            <el-popconfirm
              title="确定要删除该用户吗？此操作不可恢复！"
              confirmButtonText="确定"
              cancelButtonText="取消"
              @confirm="handleDelete(row)"
            >
              <template #reference>
                <el-button link type="danger" size="small">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>

        <el-table-column v-else label="操作" width="100" fixed="right">
          <template #default>
            <el-empty description="无权限" :image-size="40" />
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-wrapper">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :total="total"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 创建/编辑用户弹窗 -->
    <el-dialog
      v-model="dialogVisible"
      :title="dialogTitle"
      width="520px"
      :close-on-click-modal="false"
    >
      <el-form label-width="100px" :model="form">
        <el-form-item label="用户名" required>
          <el-input
            v-model="form.username"
            placeholder="请输入用户名（3-20字符）"
            maxlength="20"
            show-word-limit
            :disabled="isEdit"
          />
        </el-form-item>

        <el-form-item label="邮箱" required>
          <el-input
            v-model="form.email"
            placeholder="请输入邮箱地址"
            maxlength="50"
          />
        </el-form-item>

        <el-form-item v-if="!isEdit" label="密码" required>
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码（6-30字符）"
            maxlength="30"
            show-password
          />
        </el-form-item>

        <el-form-item v-else label="新密码">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="留空则不修改（6-30字符）"
            maxlength="30"
            show-password
          />
          <div class="form-tip">留空表示不修改密码</div>
        </el-form-item>

        <el-form-item label="角色" v-if="isAdmin">
          <el-radio-group v-model="form.role">
            <el-radio value="admin" :disabled="!isAdmin">管理员</el-radio>
            <el-radio value="operator">运维人员</el-radio>
            <el-radio value="viewer">访客</el-radio>
          </el-radio-group>
          <div v-if="!isAdmin" class="form-tip">仅管理员可分配管理员角色</div>
        </el-form-item>

        <el-form-item label="状态">
          <el-switch
            v-model="form.status"
            active-text="启用"
            inactive-text="禁用"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleSubmit">
          {{ isEdit ? '保存' : '创建' }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style lang="scss" scoped>
.page-container {
  padding: 20px;
  background: #f5f7fa;
  min-height: calc(100vh - 84px);
}

.page-header {
  margin-bottom: 16px;

  h2 {
    font-size: 22px;
    font-weight: 600;
    color: #303133;
    margin: 0;
  }
}

.toolbar-card {
  margin-bottom: 16px;

  .toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
  }
}

.table-card {
  .pagination-wrapper {
    display: flex;
    justify-content: flex-end;
    padding-top: 16px;
  }
}

.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}
</style>
