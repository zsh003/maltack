<template>
<!--  <a-descriptions bordered>-->
<!--    <a-descriptions-item v-for="(value, key) in data" :key="key" :label="key" :span="3">-->
<!--      {{ value }}-->
<!--    </a-descriptions-item>-->
<!--  </a-descriptions>-->
<!--  &lt;!&ndash;  统一卡片样式&ndash;&gt;-->
<!--  <a-card :bordered="false" class="analysis-section">-->
<!--    &lt;!&ndash; 内容 &ndash;&gt;-->
<!--  </a-card>-->
</template>

<script setup lang="ts">
defineProps<{
  data:
    | {
        filename: string
        file_size: number
        file_type: string
        mime_type: string
        md5: string
        sha1: string
        sha256: string
        analyze_time: string
      }
    | undefined
}>()
</script>

<style scoped>
.analysis-section {
  margin-bottom: 24px;
  background: #fff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.08);
}
</style>
