<template>
  <div class="file-uploader-wrapper">
    <a-col :span="20" offset="2">
      <a-card class="upload-card" title="文件上传">
        <!-- 上传组件 -->
        <a-upload-dragger
          name="file"
          action="http://localhost:5000/api/v1/analyze"
          :before-upload="beforeUpload"
          @success="handleUploadSuccess"
          @error="handleUploadError"
          @change="handleUploadChange"
          :loading="uploadLoading"
          :show-upload-list="false">
          <p class="ant-upload-drag-icon">
            <upload-outlined />
          </p>
          <p class="ant-upload-text">
            点击或拖拽文件到此处上传
          </p>
          <!-- 状态提示 -->
          <div v-if="isAnalyzing" class="analysis-status">
            正在分析中，请稍候...
          </div>
          <div v-if="uploadError" class="upload-error">
            {{ uploadError }}
          </div>
        </a-upload-dragger>
      </a-card>
    </a-col>
  </div>
</template>

<script>
import { ref } from 'vue'
import { UploadOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'

export default {
  components: {
    UploadOutlined
  },
  setup (props, { emit }) {
    const analysisResult = ref(null)
    const isAnalyzing = ref(false)
    const uploadError = ref(null)
    const uploadLoading = ref(false)

    const beforeUpload = (file) => {
      isAnalyzing.value = true // 开始分析
      uploadError.value = null // 清空错误
      analysisResult.value = null // 重置结果
      const isValidSize = file.size / 1024 / 1024 < 50
      if (!isValidSize) {
        message.error('文件大小不能超过50MB')
        return false
      }
      uploadLoading.value = true // 设置加载状态
      return true // 必须返回true才会继续上传
    }

    const handleUploadSuccess = (response, file) => {
      isAnalyzing.value = false
      uploadLoading.value = false // 关闭加载状态
      if (response && response.success) { // 假设后端返回成功标志
        analysisResult.value = response.data // 存储结果
        message.success('文件分析完成')
        emit('update-analysis-result', analysisResult.value)
      } else {
        uploadError.value = '分析失败，请检查文件格式'
        message.error(uploadError.value)
      }
    }

    const handleUploadError = (error, file) => {
      isAnalyzing.value = false
      uploadLoading.value = false // 关闭加载状态
      uploadError.value = '上传失败：' + (error.message || '未知错误')
      message.error(uploadError.value)
    }

    const handleUploadChange = (info) => {
      if (info.file.status === 'done') {
        message.success(`${info.file.name} 文件上传成功`)
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} 文件上传失败`)
      }
    }

    return {
      analysisResult,
      isAnalyzing,
      uploadError,
      uploadLoading,
      beforeUpload,
      handleUploadSuccess,
      handleUploadError,
      handleUploadChange
    }
  }
}
</script>

<style lang="less" scoped>
.file-uploader-wrapper {
  display: flex;
  padding: 20px;

  .analysis-status, .upload-error {
    text-align: center;
    margin-top: 20px;
  }
}

.upload-card {
  width: 100%;
  height: 100%;
  background: #fff;
}

.analysis-status {
  margin-top: 16px;
  color: #1890ff;
  font-weight: bold;
}

.upload-error {
  margin-top: 16px;
  color: #ff4d4f;
  font-weight: bold;
}
</style>
