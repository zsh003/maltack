<template>
  <a-layout-sider width="240" class="side-menu" :theme="theme">
    <a-menu v-model:selectedKeys="selectedKeys" mode="vertical" :theme="theme">
      <a-menu-item v-for="item in items" :key="item.key">
        {{ item.label }}
      </a-menu-item>
    </a-menu>
  </a-layout-sider>
</template>

<script setup lang="ts">
defineProps({
  theme: {
    type: String,
    default: 'light',
  },
  items: {
    type: Array,
    default: () => [],
  },
})

const selectedKeys = defineModel('selectedKeys')
</script>

<style scoped>
.side-menu {
  margin-right: 24px;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
</style>
