<template>
  <a-collapse>
    <a-collapse-panel v-for="(match, index) in data" :key="index" :header="match.title">
      <a-descriptions bordered>
        <a-descriptions-item label="ID" :span="3">
          {{ match.id }}
        </a-descriptions-item>
        <a-descriptions-item label="描述" :span="3">
          {{ match.description }}
        </a-descriptions-item>
        <a-descriptions-item label="级别" :span="3">
          <a-tag :color="getSigmaLevelType(match.level)">
            {{ match.level }}
          </a-tag>
        </a-descriptions-item>
        <a-descriptions-item label="标签" :span="3">
          <a-tag v-for="tag in match.tags" :key="tag" class="tag-margin">
            {{ tag }}
          </a-tag>
        </a-descriptions-item>
      </a-descriptions>
    </a-collapse-panel>
  </a-collapse>
</template>

<script setup lang="ts">
import { computed } from 'vue'

defineProps<{
  data:
    | Array<{
        id: string
        title: string
        description: string
        level: string
        tags: string[]
      }>
    | undefined
}>()

const getSigmaLevelType = computed(() => (level: string) => {
  switch (level.toLowerCase()) {
    case 'critical':
      return 'error'
    case 'high':
      return 'warning'
    case 'medium':
      return 'processing'
    case 'low':
      return 'success'
    default:
      return 'default'
  }
})
</script>
