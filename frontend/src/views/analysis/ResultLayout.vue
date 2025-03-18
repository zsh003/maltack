<template>
  <page-header-wrapper
    :tab-list="tabList"
    :tab-active-key="tabActiveKey"
    :tab-change="handleTabChange"
  >
    <template v-slot:content>
      <FileUploader />
    </template>
    <router-view @update-analysis-result="handleUpdateAnalysisResult" />
  </page-header-wrapper>
</template>

<script>
import FileUploader from './FileUploader.vue'

const getActiveKey = (path) => {
  switch (path) {
    case '/analysis/result/overview':
      return '1'
    case '/analysis/result/basic-info':
      return '2'
    case '/analysis/result/yara-rules':
      return '3'
    case '/analysis/result/sigma-rules':
      return '4'
    case '/analysis/result/strings':
      return '5'

    default:
      return '1'
  }
}
export default {
  components: {
    FileUploader
  },
  name: 'ResultLayout',
  data () {
    return {
      tabList: [
        { key: '1', tab: '总体一览' },
        { key: '2', tab: '文件基本信息' },
        { key: '3', tab: 'Yara规则匹配' },
        { key: '4', tab: 'Sigma规则匹配' },
        { key: '5', tab: '字符串分析' }
      ],
      tabActiveKey: '1',
      search: true,
      analysisResult: null // 存储分析结果
    }
  },
  created () {
    this.tabActiveKey = getActiveKey(this.$route.path)

    this.$watch('$route', (val) => {
      this.tabActiveKey = getActiveKey(val.path)
    })
  },
  methods: {
    handleTabChange (key) {
      this.tabActiveKey = key
      switch (key) {
        case '1':
          this.$router.push('/analysis/result/overview')
          break
        case '2':
          this.$router.push('/analysis/result/basic-info')
          break
        case '3':
          this.$router.push('/analysis/result/yara-rules')
          break
        case '4':
          this.$router.push('/analysis/result/sigma-rules')
          break
        case '5':
          this.$router.push('/analysis/result/strings')
          break
        default:
          this.$router.push('/workplace')
      }
    },
    handleUpdateAnalysisResult (result) {
      this.analysisResult = result
    }
  }
}
</script>

<style lang="less" scoped>
.analysis-status {
  margin-top: 16px;
  color: #1890ff;
  font-weight: bold;
}
</style>
