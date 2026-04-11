import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import router from '@/router'
import { post } from '@/utils/request'

export const useUserStore = defineStore(
  'user',
  () => {
    const token = ref<string>('')
    const userInfo = ref<any>({})
    const collapsed = ref<boolean>(false)

    const isLoggedIn = computed(() => !!token.value)
    const role = computed(() => userInfo.value?.role)
    const username = computed(() => userInfo.value?.username)

    async function login(loginForm: { username: string; password: string }) {
      const res: any = await post('/admin/auth/login', {
        username: loginForm.username,
        password: loginForm.password
      })

      token.value = res.access_token
      userInfo.value = {
        id: res.user.id,
        username: res.user.username,
        email: res.user.email,
        role: res.user.role
      }

      return res
    }

    function logout() {
      token.value = ''
      userInfo.value = {}
      localStorage.removeItem('token')
      router.push('/login')
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
      username,
      login,
      logout,
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
