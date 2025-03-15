<template>
  <a-tabs>
    <a-tab-pane key="ascii" tab="ASCII字符串">
      <a-table
        :dataSource="data?.ascii_strings"
        :columns="[
          { title: '偏移量', dataIndex: 'offset', key: 'offset', width: 120 },
          { title: '字符串', dataIndex: 'string', key: 'string' },
        ]"
      />
    </a-tab-pane>
    <a-tab-pane key="unicode" tab="Unicode字符串">
      <a-table
        :dataSource="data?.unicode_strings"
        :columns="[
          { title: '偏移量', dataIndex: 'offset', key: 'offset', width: 120 },
          { title: '字符串', dataIndex: 'string', key: 'string' },
        ]"
      />
    </a-tab-pane>
  </a-tabs>
</template>

<script setup lang="ts">
defineProps<{
  data:
    | {
        ascii_strings: Array<{
          offset: number
          string: string
        }>
        unicode_strings: Array<{
          offset: number
          string: string
        }>
      }
    | undefined
}>()
</script>
