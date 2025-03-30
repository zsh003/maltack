import { Card, Table, Spin, Descriptions, Tag } from 'antd';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";

const PEStaticFeatures: FC = () => {
  const { currentFileId } = useAnalysisModel();
  const [loading, setLoading] = useState(true);
  const [features, setFeatures] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get(`http://localhost:5000/api/v1/analysis/features/pe-static/${currentFileId}`);
        setFeatures(data);
      } catch (error) {
        console.error('获取PE静态特征失败', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [currentFileId]);

  if (loading) {
    return <Spin />;
  }

  const sectionColumns = [
    {
      title: '节区名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`,
    },
    {
      title: '熵值',
      dataIndex: 'entropy',
      key: 'entropy',
      render: (entropy: number) => entropy.toFixed(4),
    },
    {
      title: '虚拟大小',
      dataIndex: 'vsize',
      key: 'vsize',
      render: (vsize: number) => `${(vsize / 1024).toFixed(2)} KB`,
    },
    {
      title: '属性',
      dataIndex: 'props',
      key: 'props',
      render: (props: string[]) => (
        <>
          {props.map(prop => (
            <Tag key={prop} color="blue">{prop}</Tag>
          ))}
        </>
      ),
    },
  ];

  const exportColumns = [
    {
      title: '函数名',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '地址',
      dataIndex: 'address',
      key: 'address',
      render: (address: number) => `0x${address.toString(16)}`,
    },
    {
      title: '序号',
      dataIndex: 'ordinal',
      key: 'ordinal',
    },
  ];

  return (
    <div>
      <Card title="基本信息" style={{ marginBottom: 16 }}>
        <Descriptions bordered>
          <Descriptions.Item label="文件大小">
            {(features?.general_info?.file_size / 1024).toFixed(2)} KB
          </Descriptions.Item>
          <Descriptions.Item label="入口点">
            0x{features?.general_info?.entry_point?.toString(16)}
          </Descriptions.Item>
          <Descriptions.Item label="机器类型">
            {features?.general_info?.machine_type}
          </Descriptions.Item>
          <Descriptions.Item label="时间戳">
            {new Date(features?.general_info?.timestamp * 1000).toLocaleString()}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title="节区信息" style={{ marginBottom: 16 }}>
        <Table
          columns={sectionColumns}
          dataSource={features?.section_info?.sections}
          rowKey="name"
          pagination={false}
        />
      </Card>

      <Card title="导出函数">
        <Table
          columns={exportColumns}
          dataSource={features?.export_info?.exports}
          rowKey="name"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
          }}
        />
      </Card>
    </div>
  );
};

export default PEStaticFeatures;
