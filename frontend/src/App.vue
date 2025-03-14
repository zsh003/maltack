<script setup lang="ts">
import { ref } from 'vue'
import { UploadOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import type { UploadProps } from 'ant-design-vue'

interface AnalysisResult {
  basic_info: {
    [key: string]: string | number
  }
  pe_info?: {
    machine_type: string
    timestamp: string
    subsystem: number
    sections: Array<{
      name: string
      virtual_address: string
      virtual_size: string
      raw_size: string
    }>
    imports: Array<{
      dll: string
      functions: string[]
    }>
    exports: Array<{
      name: string
      address: string
    }>
  }
  yara_matches: Array<{
    rule_name: string
    strings: Array<{
      identifier: string
      data: string
      offset: number
    }>
    tags: string[]
    meta: Record<string, unknown>
  }>
  sigma_matches: Array<{
    id: string
    title: string
    description: string
    level: string
    tags: string[]
  }>
  string_info: {
    ascii_strings: Array<{
      offset: number
      string: string
    }>
    unicode_strings: Array<{
      offset: number
      string: string
    }>
  }
}

const analysisResult = ref<AnalysisResult | null>(null)
const activeTab = ref('basic')

const handleUploadSuccess: UploadProps['onSuccess'] = (response) => {
  analysisResult.value = response as AnalysisResult
  message.success('文件分析完成')
}

const handleUploadError: UploadProps['onError'] = () => {
  message.error('文件分析失败')
}

const beforeUpload: UploadProps['beforeUpload'] = (file) => {
  const isValidSize = file.size / 1024 / 1024 < 50
  if (!isValidSize) {
    message.error('文件大小不能超过50MB')
    return false
  }
  return true
}

const getSigmaLevelType = (level: string) => {
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
}
</script>

<template>
  <a-layout class="layout-container">
    <a-layout-header>
      <h1 class="header-title">恶意文件分析平台</h1>
    </a-layout-header>
    
    <a-layout-content class="main-content">
      <a-row :gutter="20">
        <a-col :span="8">
          <a-card class="upload-card" title="文件上传">
            <a-upload-dragger
              name="file"
              action="http://localhost:5000/api/analyze"
              :before-upload="beforeUpload"
              @success="handleUploadSuccess"
              @error="handleUploadError">
              <p class="ant-upload-drag-icon">
                <upload-outlined />
              </p>
              <p class="ant-upload-text">
                点击或拖拽文件到此处上传
              </p>
            </a-upload-dragger>
          </a-card>
        </a-col>
        
        <a-col :span="16">
          <a-card v-if="analysisResult" title="分析结果">
            <a-tabs v-model:activeKey="activeTab">
              <!-- 基础信息 -->
              <a-tab-pane key="basic" tab="基础信息">
                <a-descriptions bordered>
                  <a-descriptions-item 
                    v-for="(value, key) in analysisResult.basic_info"
                    :key="key" 
                    :label="key"
                    :span="3">
                    {{ value }}
                  </a-descriptions-item>
                </a-descriptions>
              </a-tab-pane>
              
              <!-- PE信息 -->
              <a-tab-pane key="pe" tab="PE文件信息" v-if="analysisResult.pe_info">
                <a-tabs>
                  <a-tab-pane key="basic" tab="基本信息">
                    <a-descriptions bordered>
                      <a-descriptions-item label="机器类型" :span="3">
                        {{ analysisResult.pe_info.machine_type }}
                      </a-descriptions-item>
                      <a-descriptions-item label="时间戳" :span="3">
                        {{ analysisResult.pe_info.timestamp }}
                      </a-descriptions-item>
                      <a-descriptions-item label="子系统" :span="3">
                        {{ analysisResult.pe_info.subsystem }}
                      </a-descriptions-item>
                    </a-descriptions>
                  </a-tab-pane>
                  
                  <a-tab-pane key="sections" tab="节区信息">
                    <a-table :dataSource="analysisResult.pe_info.sections" :columns="[
                      { title: '名称', dataIndex: 'name', key: 'name' },
                      { title: '虚拟地址', dataIndex: 'virtual_address', key: 'virtual_address' },
                      { title: '虚拟大小', dataIndex: 'virtual_size', key: 'virtual_size' },
                      { title: '原始大小', dataIndex: 'raw_size', key: 'raw_size' }
                    ]" />
                  </a-tab-pane>
                  
                  <a-tab-pane key="imports" tab="导入表">
                    <a-collapse>
                      <a-collapse-panel 
                        v-for="(imp, index) in analysisResult.pe_info.imports"
                        :key="index" 
                        :header="imp.dll">
                        <a-tag v-for="func in imp.functions" :key="func" class="tag-margin">
                          {{ func }}
                        </a-tag>
                      </a-collapse-panel>
                    </a-collapse>
                  </a-tab-pane>
                  
                  <a-tab-pane key="exports" tab="导出表">
                    <a-table :dataSource="analysisResult.pe_info.exports" :columns="[
                      { title: '函数名', dataIndex: 'name', key: 'name' },
                      { title: '地址', dataIndex: 'address', key: 'address' }
                    ]" />
                  </a-tab-pane>
                </a-tabs>
              </a-tab-pane>
              
              <!-- YARA规则匹配 -->
              <a-tab-pane key="yara" tab="YARA规则匹配">
                <a-collapse>
                  <a-collapse-panel 
                    v-for="(match, index) in analysisResult.yara_matches"
                    :key="index"
                    :header="match.rule_name">
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
                        <a-table :dataSource="match.strings" :columns="[
                          { title: '标识符', dataIndex: 'identifier', key: 'identifier' },
                          { title: '数据', dataIndex: 'data', key: 'data' },
                          { title: '偏移量', dataIndex: 'offset', key: 'offset' }
                        ]" />
                      </a-descriptions-item>
                    </a-descriptions>
                  </a-collapse-panel>
                </a-collapse>
              </a-tab-pane>
              
              <!-- SIGMA规则匹配 -->
              <a-tab-pane key="sigma" tab="SIGMA规则匹配">
                <a-collapse>
                  <a-collapse-panel 
                    v-for="(match, index) in analysisResult.sigma_matches"
                    :key="index"
                    :header="match.title">
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
              </a-tab-pane>
              
              <!-- 字符串分析 -->
              <a-tab-pane key="strings" tab="字符串分析">
                <a-tabs>
                  <a-tab-pane key="ascii" tab="ASCII字符串">
                    <a-table :dataSource="analysisResult.string_info.ascii_strings" :columns="[
                      { title: '偏移量', dataIndex: 'offset', key: 'offset', width: 120 },
                      { title: '字符串', dataIndex: 'string', key: 'string' }
                    ]" />
                  </a-tab-pane>
                  <a-tab-pane key="unicode" tab="Unicode字符串">
                    <a-table :dataSource="analysisResult.string_info.unicode_strings" :columns="[
                      { title: '偏移量', dataIndex: 'offset', key: 'offset', width: 120 },
                      { title: '字符串', dataIndex: 'string', key: 'string' }
                    ]" />
                  </a-tab-pane>
                </a-tabs>
              </a-tab-pane>
            </a-tabs>
          </a-card>
        </a-col>
      </a-row>
    </a-layout-content>
  </a-layout>
</template>

<style scoped>
.layout-container {
  min-height: 100vh;
}

.header-title {
  color: white;
  text-align: center;
  margin: 0;
}

.main-content {
  padding: 20px;
}

.upload-card {
  margin-bottom: 20px;
}

.tag-margin {
  margin: 2px;
}

pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  margin: 0;
  padding: 10px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

:deep(.ant-layout-header) {
  background: #1890ff;
  padding: 0;
  line-height: 64px;
}

:deep(.ant-card-head-title) {
  font-size: 16px;
  font-weight: 500;
}

:deep(.ant-upload-drag) {
  padding: 16px;
}

:deep(.ant-descriptions-item-label) {
  width: 120px;
}
</style>
