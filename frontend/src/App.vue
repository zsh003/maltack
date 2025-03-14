<script setup lang="ts">
import { ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const analysisResult = ref(null)
const activeTab = ref('basic')
const activeYaraNames = ref([])
const activeSigmaNames = ref([])

const handleUploadSuccess = (response: any) => {
  analysisResult.value = response
  ElMessage.success('文件分析完成')
}

const handleUploadError = () => {
  ElMessage.error('文件分析失败')
}

const beforeUpload = (file: File) => {
  const isValidSize = file.size / 1024 / 1024 < 50
  if (!isValidSize) {
    ElMessage.error('文件大小不能超过50MB')
    return false
  }
  return true
}

const getSigmaLevelType = (level: string) => {
  switch (level.toLowerCase()) {
    case 'critical':
      return 'danger'
    case 'high':
      return 'warning'
    case 'medium':
      return 'info'
    case 'low':
      return 'success'
    default:
      return ''
  }
}
</script>

<template>
  <el-container class="layout-container">
    <el-header>
      <h1>恶意文件分析平台</h1>
    </el-header>
    
    <el-main>
      <el-row :gutter="20">
        <el-col :span="8">
          <el-card class="upload-card">
            <template #header>
              <div class="card-header">
                <span>文件上传</span>
              </div>
            </template>
            <el-upload
              class="upload-demo"
              drag
              action="http://localhost:5000/api/analyze"
              :on-success="handleUploadSuccess"
              :on-error="handleUploadError"
              :before-upload="beforeUpload">
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">
                拖拽文件到此处或 <em>点击上传</em>
              </div>
            </el-upload>
          </el-card>
        </el-col>
        
        <el-col :span="16">
          <el-card v-if="analysisResult">
            <template #header>
              <div class="card-header">
                <span>分析结果</span>
              </div>
            </template>
            
            <el-tabs v-model="activeTab">
              <!-- 基础信息 -->
              <el-tab-pane label="基础信息" name="basic">
                <el-descriptions :column="1" border>
                  <el-descriptions-item v-for="(value, key) in analysisResult.basic_info"
                    :key="key" :label="key">{{ value }}</el-descriptions-item>
                </el-descriptions>
              </el-tab-pane>
              
              <!-- PE信息 -->
              <el-tab-pane label="PE文件信息" name="pe" v-if="analysisResult.pe_info">
                <el-tabs>
                  <el-tab-pane label="基本信息">
                    <el-descriptions :column="1" border>
                      <el-descriptions-item label="机器类型">
                        {{ analysisResult.pe_info.machine_type }}
                      </el-descriptions-item>
                      <el-descriptions-item label="时间戳">
                        {{ analysisResult.pe_info.timestamp }}
                      </el-descriptions-item>
                      <el-descriptions-item label="子系统">
                        {{ analysisResult.pe_info.subsystem }}
                      </el-descriptions-item>
                    </el-descriptions>
                  </el-tab-pane>
                  
                  <el-tab-pane label="节区信息">
                    <el-table :data="analysisResult.pe_info.sections">
                      <el-table-column prop="name" label="名称" />
                      <el-table-column prop="virtual_address" label="虚拟地址" />
                      <el-table-column prop="virtual_size" label="虚拟大小" />
                      <el-table-column prop="raw_size" label="原始大小" />
                    </el-table>
                  </el-tab-pane>
                  
                  <el-tab-pane label="导入表">
                    <el-collapse>
                      <el-collapse-item v-for="(imp, index) in analysisResult.pe_info.imports"
                        :key="index" :title="imp.dll">
                        <el-tag v-for="func in imp.functions" :key="func"
                          style="margin: 2px">{{ func }}</el-tag>
                      </el-collapse-item>
                    </el-collapse>
                  </el-tab-pane>
                  
                  <el-tab-pane label="导出表">
                    <el-table :data="analysisResult.pe_info.exports">
                      <el-table-column prop="name" label="函数名" />
                      <el-table-column prop="address" label="地址" />
                    </el-table>
                  </el-tab-pane>
                </el-tabs>
              </el-tab-pane>
              
              <!-- YARA规则匹配 -->
              <el-tab-pane label="YARA规则匹配" name="yara">
                <el-collapse v-model="activeYaraNames">
                  <el-collapse-item v-for="(match, index) in analysisResult.yara_matches"
                    :key="index" :title="match.rule_name">
                    <el-descriptions :column="1" border>
                      <el-descriptions-item label="标签">
                        <el-tag v-for="tag in match.tags" :key="tag" style="margin: 2px">{{ tag }}</el-tag>
                      </el-descriptions-item>
                      <el-descriptions-item label="元数据">
                        <pre>{{ JSON.stringify(match.meta, null, 2) }}</pre>
                      </el-descriptions-item>
                      <el-descriptions-item label="匹配字符串">
                        <el-table :data="match.strings">
                          <el-table-column prop="identifier" label="标识符" />
                          <el-table-column prop="data" label="数据" />
                          <el-table-column prop="offset" label="偏移量" />
                        </el-table>
                      </el-descriptions-item>
                    </el-descriptions>
                  </el-collapse-item>
                </el-collapse>
              </el-tab-pane>
              
              <!-- SIGMA规则匹配 -->
              <el-tab-pane label="SIGMA规则匹配" name="sigma">
                <el-collapse v-model="activeSigmaNames">
                  <el-collapse-item v-for="(match, index) in analysisResult.sigma_matches"
                    :key="index" :title="match.title">
                    <el-descriptions :column="1" border>
                      <el-descriptions-item label="ID">{{ match.id }}</el-descriptions-item>
                      <el-descriptions-item label="描述">{{ match.description }}</el-descriptions-item>
                      <el-descriptions-item label="级别">
                        <el-tag :type="getSigmaLevelType(match.level)">{{ match.level }}</el-tag>
                      </el-descriptions-item>
                      <el-descriptions-item label="标签">
                        <el-tag v-for="tag in match.tags" :key="tag" style="margin: 2px">{{ tag }}</el-tag>
                      </el-descriptions-item>
                    </el-descriptions>
                  </el-collapse-item>
                </el-collapse>
              </el-tab-pane>
              
              <!-- 字符串分析 -->
              <el-tab-pane label="字符串分析" name="strings">
                <el-tabs>
                  <el-tab-pane label="ASCII字符串">
                    <el-table :data="analysisResult.string_info.ascii_strings">
                      <el-table-column prop="offset" label="偏移量" width="120" />
                      <el-table-column prop="string" label="字符串" />
                    </el-table>
                  </el-tab-pane>
                  <el-tab-pane label="Unicode字符串">
                    <el-table :data="analysisResult.string_info.unicode_strings">
                      <el-table-column prop="offset" label="偏移量" width="120" />
                      <el-table-column prop="string" label="字符串" />
                    </el-table>
                  </el-tab-pane>
                </el-tabs>
              </el-tab-pane>
            </el-tabs>
          </el-card>
        </el-col>
      </el-row>
    </el-main>
  </el-container>
</template>

<style scoped>
.layout-container {
  min-height: 100vh;
}

.el-header {
  background-color: #409EFF;
  color: white;
  text-align: center;
  line-height: 60px;
}

.el-main {
  padding: 20px;
}

.upload-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.el-tag {
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
</style>
