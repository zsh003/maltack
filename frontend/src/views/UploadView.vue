<template>
  <a-row :gutter="24" class="gap-md">
    <!-- 上传区域 -->
    <a-col :span="6">
      <a-card class="upload-card" title="文件上传" :bordered="false">
        <a-upload-dragger
          name="file"
          :disabled="isLoading"
          :action="uploadUrl"
          :before-upload="beforeUpload"
          @success="handleUploadSuccess"
          @error="handleUploadError"
        >
          <div v-if="isLoading" class="upload-loading">
            <a-spin size="large" />
          </div>
          <div class="upload-icon">
            <upload-outlined />
          </div>
          <p class="upload-text">点击或拖拽文件到此处上传</p>
          <div class="upload-tip">支持格式：.exe, .dll, .elf</div>
        </a-upload-dragger>
      </a-card>
    </a-col>
  </a-row>
</template>
<script setup lang="ts">
import { useStore } from 'vuex'

import { ref } from 'vue'
import { message } from 'ant-design-vue'
import type { UploadProps } from 'ant-design-vue'

const uploadUrl = 'http://localhost:5000/api/v1/analyze'
import { UploadOutlined } from '@ant-design/icons-vue'

const isLoading = ref(false)

const store = useStore()
import { useRouter } from 'vue-router'
const router = useRouter()

const handleUploadSuccess: UploadProps['onSuccess'] = (response) => {
  // analysisResult.value = response as AnalysisResult;
  store.dispatch('updateAnalysisResult', response)
  message.success('文件分析完成')
  isLoading.value = false // 关闭加载状态
  router.push('/analysis')
}

const handleUploadError: UploadProps['onError'] = () => {
  message.error('文件分析失败')
  isLoading.value = false // 关闭加载状态
}

const beforeUpload: UploadProps['beforeUpload'] = (file) => {
  const isValidSize = file.size / 1024 / 1024 < 50
  isLoading.value = true // 开始上传时显示加载
  if (!isValidSize) {
    message.error('文件大小不能超过50MB')
    isLoading.value = false // 关闭加载状态
    return false
  }
  return true
}
</script>
<style scoped>
/* 上传卡片样式 */
.upload-card {
  background: #fff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.08);
}

.upload-icon {
  font-size: 48px;
  color: #1890ff;
  margin-bottom: 16px;
}
/* 加载样式 */
.upload-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.upload-text {
  font-size: 16px;
  color: #666;
  margin-bottom: 8px;
}

.upload-tip {
  color: #999;
  font-size: 12px;
}
</style>
