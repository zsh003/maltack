<template v-slot:content>
  <div class="ant-pro-page-header-search">
    <a-input-search size="large" style="width: 80%; max-width: 522px;">
      <template v-slot:enterButton>
        搜索
      </template>
    </a-input-search>
  </div>
</template>

<script setup>

</script>

<style scoped lang="less">
.ant-pro-page-header-search {
  text-align: center;
  margin-bottom: 16px;
}
</style>
