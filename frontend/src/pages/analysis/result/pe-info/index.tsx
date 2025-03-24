import { useRequest } from '@umijs/max';
import { Card, Table, Descriptions, Spin, Tabs } from 'antd';
import type { FC } from 'react';
import { getPEInfo } from '../service';

const { TabPane } = Tabs;

const PEInfo: FC = () => {
  const { data, loading } = useRequest(() => getPEInfo());

  if (loading) {
    return <Spin />;
  }

  const peInfo = data?.pe_info || {};

  const sectionColumns = [
    {
      title: '节区名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '虚拟地址',
      dataIndex: 'virtual_address',
      key: 'virtual_address',
    },
    {
      title: '虚拟大小',
      dataIndex: 'virtual_size',
      key: 'virtual_size',
    },
    {
      title: '原始大小',
      dataIndex: 'raw_size',
      key: 'raw_size',
    },
  ];

  const importColumns = [
    {
      title: 'DLL名称',
      dataIndex: 'dll',
      key: 'dll',
    },
    {
      title: '导入函数',
      dataIndex: 'functions',
      key: 'functions',
      render: (functions: string[]) => functions.join(', '),
    },
  ];

  const exportColumns = [
    {
      title: '导出函数名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '地址',
      dataIndex: 'address',
      key: 'address',
    },
  ];

  return (
    <Card title="PE文件信息">
      <Descriptions column={2} bordered>
        <Descriptions.Item label="机器类型">{peInfo.machine_type}</Descriptions.Item>
        <Descriptions.Item label="时间戳">{peInfo.timestamp}</Descriptions.Item>
        <Descriptions.Item label="子系统">{peInfo.subsystem}</Descriptions.Item>
        <Descriptions.Item label="DLL特征">{peInfo.dll_characteristics}</Descriptions.Item>
      </Descriptions>

      <Tabs defaultActiveKey="sections" style={{ marginTop: 16 }}>
        <TabPane tab="节区信息" key="sections">
          <Table
            columns={sectionColumns}
            dataSource={peInfo.sections || []}
            rowKey="name"
            pagination={false}
          />
        </TabPane>
        <TabPane tab="导入表" key="imports">
          <Table
            columns={importColumns}
            dataSource={peInfo.imports || []}
            rowKey="dll"
            pagination={false}
          />
        </TabPane>
        <TabPane tab="导出表" key="exports">
          <Table
            columns={exportColumns}
            dataSource={peInfo.exports || []}
            rowKey="name"
            pagination={false}
          />
        </TabPane>
      </Tabs>
    </Card>
  );
};

export default PEInfo;
