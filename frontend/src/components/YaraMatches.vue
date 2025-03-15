<template>
  <a-collapse>
    <a-collapse-panel v-for="(match, index) in data" :key="index" :header="match.rule_name">
      <a-descriptions bordered>
        <a-descriptions-item label="标签" :span="3">
          <a-tag v-for="tag in match.tags" :key="tag" class="tag-margin">
            {{ tag }}
          </a-tag>
        </a-descriptions-item>
        <a-descriptions-item label="元数据" :span="3">
          <pre>{{ JSON.stringify(match.meta, null, 2) }}</pre>
        </a-descriptions-item>
        <a-descriptions-item label="匹配字符串" :span="3">
          <a-table
            :dataSource="match.strings"
            :columns="[
              { title: '标识符', dataIndex: 'identifier', key: 'identifier' },
              { title: '数据', dataIndex: 'data', key: 'data' },
              { title: '偏移量', dataIndex: 'offset', key: 'offset' },
            ]"
          />
        </a-descriptions-item>
      </a-descriptions>
    </a-collapse-panel>
  </a-collapse>
</template>

<script setup lang="ts">
defineProps<{
  data:
    | Array<{
        rule_name: string
        strings: Array<{
          identifier: string
          data: string
          offset: number
        }>
        tags: string[]
        meta: Record<string, unknown>
      }>
    | undefined
}>()
</script>
