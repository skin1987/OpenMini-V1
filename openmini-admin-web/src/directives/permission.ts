import type { Directive, DirectiveBinding } from 'vue'
import { useUserStore } from '@/stores/user'

const permission: Directive = {
  mounted(el: HTMLElement, binding: DirectiveBinding<string[]>) {
    const { value } = binding

    if (value && Array.isArray(value) && value.length > 0) {
      const userStore = useUserStore()
      const currentRole = userStore.role

      if (currentRole) {
        const hasPermission = value.includes(currentRole)

        if (!hasPermission) {
          el.parentNode?.removeChild(el)
        }
      } else {
        el.parentNode?.removeChild(el)
      }
    } else {
      throw new Error('需要指定权限角色，例如 v-permission="[\'admin\', \'operator\']"')
    }
  }
}

export default permission
