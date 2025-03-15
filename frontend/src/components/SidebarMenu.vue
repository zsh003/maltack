<template>
  <a-menu mode="inline" :default-selected-keys="['basic']">
    <a-menu-item key="basic">
      <template #icon><profile-outlined /></template>
      基础信息
    </a-menu-item>
    <a-menu-item key="rules">
      <template #icon><security-scan-outlined /></template>
      规则匹配
    </a-menu-item>
    <a-menu-item key="strings">
      <template #icon><file-text-outlined /></template>
      字符串分析
    </a-menu-item>
  </a-menu>
</template>

<script setup lang="ts">
import { ProfileOutlined, SecurityScanOutlined, FileTextOutlined } from '@ant-design/icons-vue'
</script>
