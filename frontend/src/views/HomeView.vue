<template>
  <default-layout>
    <a-row :gutter="20">
      <a-col :span="6">
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
    </a-row>
  </default-layout>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { UploadOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';
import type { UploadProps } from 'ant-design-vue';
import DefaultLayout from '../layouts/DefaultLayout.vue';
import BasicInfo from '../components/BasicInfo.vue';
import PeInfo from '../components/PeInfo.vue';
import YaraMatches from '../components/YaraMatches.vue';
import SigmaMatches from '../components/SigmaMatches.vue';
import StringAnalysis from '../components/StringAnalysis.vue';

interface AnalysisResult {
  basic_info: {
    filename: string;
    file_size: number;
    file_type: string;
    mime_type: string;
    md5: string;
    sha1: string;
    sha256: string;
    analyze_time: string;
  };
  pe_info?: {
    machine_type: string;
    timestamp: string;
    subsystem: number;
    dll_characteristics: number;
    sections: Array<{
      name: string;
      virtual_address: string;
      virtual_size: string;
      raw_size: string;
    }>;
    imports: Array<{
      dll: string;
      functions: string[];
    }>;
    exports: Array<{
      name: string;
      address: string;
    }>;
  };
  yara_matches: Array<{
    rule_name: string;
    strings: Array<{
      identifier: string;
      data: string;
      offset: number;
    }>;
    tags: string[];
    meta: Record<string, unknown>;
  }>;
  sigma_matches: Array<{
    id: string;
    title: string;
    description: string;
    level: string;
    tags: string[];
  }>;
  string_info: {
    ascii_strings: Array<{
      offset: number;
      string: string;
    }>;
    unicode_strings: Array<{
      offset: number;
      string: string;
    }>;
  };
}

const analysisResult = ref<AnalysisResult | null>(null);
const activeTab = ref('basic');

const handleUploadSuccess: UploadProps['onSuccess'] = (response) => {
  analysisResult.value = response as AnalysisResult;
  message.success('文件分析完成');
};

const handleUploadError: UploadProps['onError'] = () => {
  message.error('文件分析失败');
};

const beforeUpload: UploadProps['beforeUpload'] = (file) => {
  const isValidSize = file.size / 1024 / 1024 < 50;
  if (!isValidSize) {
    message.error('文件大小不能超过50MB');
    return false;
  }
  return true;
};
</script>



