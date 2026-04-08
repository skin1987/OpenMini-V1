import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import router from '@/router'
import type { LoginForm, LoginResponse, UserInfo } from '@/types'
import { UserRole } from '@/types'

export const useUserStore = defineStore(
  'user',
  () => {
    const token = ref<string>('')
    const userInfo = ref<Partial<UserInfo>>({})
    const collapsed = ref<boolean>(false)

    const isLoggedIn = computed(() => !!token.value)
    const role = computed(() => userInfo.value.role)

    async function login(loginForm: LoginForm): Promise<LoginResponse> {
      return new Promise((resolve, reject) => {
        setTimeout(() => {
          try {
            const mockData: LoginResponse = {
              access_token: 'mock-jwt-token-' + Date.now(),
              refresh_token: 'mock-refresh-token-' + Date.now(),
              expires_in: 7200,
              user_info: {
                id: 1,
                username: loginForm.username,
                email: `${loginForm.username}@openmini.com`,
                role: UserRole.ADMIN,
                avatar: '',
                last_login_at: new Date().toISOString(),
                created_at: '2025-01-01T00:00:00Z'
              }
            }

            token.value = mockData.access_token
            userInfo.value = mockData.user_info
            localStorage.setItem('token', mockData.access_token)

            resolve(mockData)
          } catch (error) {
            reject(error)
          }
        }, 800)
      })
    }

    function logout() {
      token.value = ''
      userInfo.value = {}
      localStorage.removeItem('token')
      router.push('/login')
    }

    async function getUserInfo(): Promise<UserInfo> {
      return new Promise((resolve) => {
        setTimeout(() => {
          if (userInfo.value.id) {
            resolve(userInfo.value as UserInfo)
          } else {
            const mockUserInfo: UserInfo = {
              id: 1,
              username: 'admin',
              email: 'admin@openmini.com',
              role: UserRole.ADMIN,
              avatar: '',
              last_login_at: new Date().toISOString(),
              created_at: '2025-01-01T00:00:00Z'
            }
            userInfo.value = mockUserInfo
            resolve(mockUserInfo)
          }
        }, 300)
      })
    }

    function resetState() {
      token.value = ''
      userInfo.value = {}
      collapsed.value = false
      localStorage.removeItem('token')
    }

    return {
      token,
      userInfo,
      collapsed,
      isLoggedIn,
      role,
      login,
      logout,
      getUserInfo,
      resetState
    }
  },
  {
    persist: {
      key: 'user-store',
      storage: localStorage,
      paths: ['token', 'userInfo']
    }
  } as any
)
