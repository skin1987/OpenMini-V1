<script setup lang="ts">
interface Props {
  status: 'success' | 'warning' | 'danger' | 'info'
  text: string
}

defineProps<Props>()

const typeMap = {
  success: 'success',
  warning: 'warning',
  danger: 'danger',
  info: 'info'
}
</script>

<template>
  <el-tag :type="typeMap[status]" effect="dark" size="small">
    {{ text }}
  </el-tag>
</template>
