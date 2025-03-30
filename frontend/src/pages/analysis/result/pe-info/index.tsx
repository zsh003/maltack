import { Card, Table, Descriptions, Spin, Tabs, Input, Tooltip, message } from 'antd';
import type { FC } from 'react';
import { useState, useEffect, useMemo } from 'react';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";

const { TabPane } = Tabs;
const { Search } = Input;

// 机器类型映射表
const MACHINE_TYPE_MAP: { [key: number]: string } = {
  0x14c: 'Intel 386',
  0x8664: 'x64',
  0x1c0: 'ARM',
};

// 时间戳转换
const formatTimestamp = (timestamp: number) => {
  try {
    return new Date(timestamp * 1000).toLocaleString();
  } catch (e) {
    return '无效时间戳';
  }
};

const PEInfo: FC = () => {
  const { currentFileId } = useAnalysisModel();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any>(null);
  const [searchParams, setSearchParams] = useState({
    sections: '',
    imports: '',
    exports: ''
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get(`http://localhost:5000/api/v1/analysis/result/pe-info/${currentFileId}`);
        setData(data);
      } catch (error) {
        console.error('获取PE信息失败', error);
        message.error('获取PE信息失败，请稍后重试');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [currentFileId]);

  // 使用useMemo优化性能
  const sectionColumns = useMemo(() => [
    {
      title: '节区名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '虚拟地址',
      dataIndex: 'virtual_address',
      key: 'virtual_address',
      render: (val: number) => `0x${val.toString(16).toUpperCase()}`,
    },
    {
      title: '虚拟大小',
      dataIndex: 'virtual_size',
      key: 'virtual_size',
      render: (val: number) => `${(val / 1024).toFixed(2)} KB`,
    },
    {
      title: '原始大小',
      dataIndex: 'raw_size',
      key: 'raw_size',
      render: (val: number) => `${(val / 1024).toFixed(2)} KB`,
    },
  ], []);

  const importColumns = useMemo(() => [
    {
      title: 'DLL名称',
      dataIndex: 'dll',
      key: 'dll',
    },
    {
      title: '导入函数',
      dataIndex: 'functions',
      key: 'functions',
      render: (functions: string[]) => (
        <Tooltip title={functions.join(', ')}>
          <span>{functions.slice(0, 3).join(', ')}{functions.length > 3 && '...'}</span>
        </Tooltip>
      ),
    },
  ], []);

  const exportColumns = useMemo(() => [
    {
      title: '导出函数名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '地址',
      dataIndex: 'address',
      key: 'address',
      render: (val: number) => `0x${val.toString(16).toUpperCase()}`,
    },
  ], []);

  // 过滤函数
  const filterData = (dataSource: any[], key: keyof typeof searchParams) => {
    return dataSource?.filter(item =>
      Object.values(item).some(value =>
        String(value).toLowerCase().includes(searchParams[key].toLowerCase())
      )
    ) || [];
  };

  if (loading) {
    return <Spin tip="正在加载PE信息..." style={{ marginTop: 20 }} />;
  }

  const peInfo = data?.pe_info || {};

  return (
    <Card title="PE文件分析报告" bordered={false}>
      <Descriptions
        column={2}
        bordered
        labelStyle={{ fontWeight: 'bold', width: 120 }}
      >
        <Descriptions.Item label="机器类型">
          {MACHINE_TYPE_MAP[peInfo.machine_type] || peInfo.machine_type}
        </Descriptions.Item>
        <Descriptions.Item label="编译时间">
          {formatTimestamp(peInfo.timestamp)}
        </Descriptions.Item>
        <Descriptions.Item label="子系统">
          {peInfo.subsystem}
        </Descriptions.Item>
        <Descriptions.Item label="DLL特征">
          {peInfo.dll_characteristics}
        </Descriptions.Item>
      </Descriptions>

      <Tabs defaultActiveKey="sections" style={{ marginTop: 16 }}>
        <TabPane tab="节区信息" key="sections">
          <Search
            placeholder="搜索节区"
            onChange={e => setSearchParams(p => ({...p, sections: e.target.value}))}
            style={{ width: 300, marginBottom: 16 }}
          />
          <Table
            columns={sectionColumns}
            dataSource={filterData(peInfo.sections, 'sections')}
            rowKey="name"
            pagination={{ pageSize: 10 }}
            scroll={{ x: true }}
          />
        </TabPane>

        <TabPane tab="导入表" key="imports">
          <Search
            placeholder="搜索DLL或函数"
            onChange={e => setSearchParams(p => ({...p, imports: e.target.value}))}
            style={{ width: 300, marginBottom: 16 }}
          />
          <Table
            columns={importColumns}
            dataSource={filterData(peInfo.imports, 'imports')}
            rowKey="dll"
            pagination={{ pageSize: 10 }}
            scroll={{ x: true }}
          />
        </TabPane>

        <TabPane tab="导出表" key="exports">
          <Search
            placeholder="搜索导出函数"
            onChange={e => setSearchParams(p => ({...p, exports: e.target.value}))}
            style={{ width: 300, marginBottom: 16 }}
          />
          <Table
            columns={exportColumns}
            dataSource={filterData(peInfo.exports, 'exports')}
            rowKey="name"
            pagination={{ pageSize: 10 }}
            scroll={{ x: true }}
          />
        </TabPane>
      </Tabs>
    </Card>
  );
};

export default PEInfo;
