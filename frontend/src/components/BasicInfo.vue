<template>
  <a-descriptions bordered>
    <a-descriptions-item
        v-for="(value, key) in data"
        :key="key"
        :label="key"
        :span="3">
      {{ value }}
    </a-descriptions-item>
  </a-descriptions>
</template>

<script setup lang="ts">
defineProps<{ data: {
    filename: string;
    file_size: number;
    file_type: string;
    mime_type: string;
    md5: string;
    sha1: string;
    sha256: string;
    analyze_time: string;
  } | undefined }>();
</script>



