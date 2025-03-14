<template>
  <a-layout style="min-height: 100vh;">
    <a-layout-sider collapsible v-model:collapsed="collapsed" width="256">
      <div class="logo" />
      <a-menu theme="dark" mode="inline" :default-selected-keys="['1']">
        <a-menu-item key="1">
          <pie-chart-outlined />
          <span>首页</span>
        </a-menu-item>
        <!-- Add more menu items here -->
      </a-menu>
    </a-layout-sider>
    <a-layout>
      <a-layout-header style="background: #fff; padding: 0;" />
      <a-layout-content style="margin: 24px 16px; padding: 24px; background: #fff; min-height: 280px;">
        <slot />
      </a-layout-content>
      <a-layout-footer style="text-align: center;">
        Malware Analysis Platform ©2023 Created by Your Name
      </a-layout-footer>
    </a-layout>
  </a-layout>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { PieChartOutlined } from '@ant-design/icons-vue';

const collapsed = ref(false);
</script>

<style scoped>
.logo {
  height: 32px;
  margin: 16px;
  background: rgba(255, 255, 255, 0.3);
}
</style>



