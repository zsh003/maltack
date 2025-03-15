<template>
  <!-- 分析结果区域 -->
  <a-col :span="18">
    <a-card class="analysis-card" :bordered="false">
      <a-tabs v-model:activeKey="activeTab" size="large">
        <a-col :span="18">
          <a-tabs v-model:activeKey="activeTab">
            <a-tab-pane key="basic" tab="基础信息">
              <basic-info :data="analysisResult?.basic_info" />
            </a-tab-pane>

            <a-tab-pane key="pe" tab="PE文件信息" v-if="analysisResult?.pe_info">
              <pe-info :data="analysisResult?.pe_info" />
            </a-tab-pane>

            <a-tab-pane key="yara" tab="YARA规则匹配">
              <yara-matches :data="analysisResult?.yara_matches" />
            </a-tab-pane>

            <a-tab-pane key="sigma" tab="SIGMA规则匹配">
              <sigma-matches :data="analysisResult?.sigma_matches" />
            </a-tab-pane>

            <a-tab-pane key="strings" tab="字符串分析">
              <string-analysis :data="analysisResult?.string_info" />
            </a-tab-pane>
          </a-tabs>
        </a-col>
      </a-tabs>
    </a-card>
  </a-col>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import BasicInfo from '../components/BasicInfo.vue'
import PeInfo from '../components/PeInfo.vue'
import YaraMatches from '../components/YaraMatches.vue'
import SigmaMatches from '../components/SigmaMatches.vue'
import StringAnalysis from '../components/StringAnalysis.vue'

const analysisResult = ref<AnalysisResult | null>(null)
const activeTab = ref('basic')
</script>

<style scoped>
/* 分析结果卡片 */
.analysis-card {
  min-height: 80vh;
  background: #fff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.08);
}

/* 标签页间距优化 */
.ant-tabs-large .ant-tabs-tab {
  margin-right: 24px;
}
</style>
