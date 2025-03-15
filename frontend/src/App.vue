<template>
  <BasicLayout />
</template>

<script setup lang="ts">
import BasicLayout from './layouts/BasicLayout.vue'

// const BasicLayout = require('./layouts/BasicLayout.vue');
</script>

<style>
/* 在 App.vue 的 <style> 标签内 */
#app {
  font-family: 'Inter', sans-serif; /* 更现代的字体 */
  color: #1a1a1a; /* 深色文字更易读 */
  padding: 24px; /* 内容与边缘保持距离 */
}

/* 新增全局间距类 */
.gap-md {
  gap: 24px;
}

/* 优化标签间距 */
.tag-margin {
  margin-right: 12px; /* 减少间距，避免过宽 */
}
</style>
