<template>
  <a-tabs>
    <a-tab-pane key="basic" tab="基本信息">
      <a-descriptions bordered>
        <a-descriptions-item label="机器类型" :span="3">
          {{ data?.machine_type }}
        </a-descriptions-item>
        <a-descriptions-item label="时间戳" :span="3">
          {{ data?.timestamp }}
        </a-descriptions-item>
        <a-descriptions-item label="子系统" :span="3">
          {{ data?.subsystem }}
        </a-descriptions-item>
        <a-descriptions-item label="DLL特性" :span="3">
          {{ data?.dll_characteristics }}
        </a-descriptions-item>
      </a-descriptions>
    </a-tab-pane>

    <a-tab-pane key="sections" tab="节区信息">
      <a-table :dataSource="data?.sections" :columns="[
        { title: '名称', dataIndex: 'name', key: 'name' },
        { title: '虚拟地址', dataIndex: 'virtual_address', key: 'virtual_address' },
        { title: '虚拟大小', dataIndex: 'virtual_size', key: 'virtual_size' },
        { title: '原始大小', dataIndex: 'raw_size', key: 'raw_size' }
      ]" />
    </a-tab-pane>

    <a-tab-pane key="imports" tab="导入表">
      <a-collapse>
        <a-collapse-panel
            v-for="(imp, index) in data?.imports"
            :key="index"
            :header="imp.dll">
          <a-tag v-for="func in imp.functions" :key="func" class="tag-margin">
            {{ func }}
          </a-tag>
        </a-collapse-panel>
      </a-collapse>
    </a-tab-pane>

    <a-tab-pane key="exports" tab="导出表">
      <a-table :dataSource="data?.exports" :columns="[
        { title: '函数名', dataIndex: 'name', key: 'name' },
        { title: '地址', dataIndex: 'address', key: 'address' }
      ]" />
    </a-tab-pane>
  </a-tabs>
</template>

<script setup lang="ts">
defineProps<{ data: {
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
  } | undefined }>();
</script>



