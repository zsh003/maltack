import { Card, Table, Spin, Tag, Descriptions } from 'antd';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";

const FeatureEngineering: FC = () => {
  const { currentFileId } = useAnalysisModel();
  const [loading, setLoading] = useState(true);
  const [features, setFeatures] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get(`http://localhost:5000/api/v1/analysis/features/engineering/${currentFileId}`);
        setFeatures(data);
      } catch (error) {
        console.error('获取特征工程数据失败', error);
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
      title: '特征名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '值',
      dataIndex: 'value',
      key: 'value',
      render: (value: any) => {
        if (typeof value === 'boolean') {
          return <Tag color={value ? 'green' : 'red'}>{value ? '是' : '否'}</Tag>;
        }
        return value;
      },
    },
  ];

  const stringMatchColumns = [
    {
      title: '模式类型',
      dataIndex: 'type',
      key: 'type',
    },
    {
      title: '匹配数量',
      dataIndex: 'count',
      key: 'count',
    },
    {
      title: '示例',
      dataIndex: 'examples',
      key: 'examples',
      render: (examples: string[]) => (
        <>
          {examples.slice(0, 3).map((example, index) => (
            <Tag key={index}>{example}</Tag>
          ))}
        </>
      ),
    },
  ];

  const yaraColumns = [
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '匹配结果',
      dataIndex: 'matched',
      key: 'matched',
      render: (matched: boolean) => (
        <Tag color={matched ? 'red' : 'green'}>
          {matched ? '匹配' : '不匹配'}
        </Tag>
      ),
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
  ];

  const opcodeColumns = [
    {
      title: '操作码',
      dataIndex: 'opcode',
      key: 'opcode',
    },
    {
      title: '出现次数',
      dataIndex: 'count',
      key: 'count',
    },
    {
      title: '频率',
      dataIndex: 'frequency',
      key: 'frequency',
      render: (frequency: number) => `${(frequency * 100).toFixed(2)}%`,
    },
  ];

  return (
    <div>
      <Card title="节区信息特征" style={{ marginBottom: 16 }}>
        <Table
          columns={sectionColumns}
          dataSource={Object.entries(features?.section_info || {}).map(([name, value]) => ({
            name,
            value,
          }))}
          rowKey="name"
          pagination={false}
        />
      </Card>

      <Card title="字符串模式匹配" style={{ marginBottom: 16 }}>
        <Table
          columns={stringMatchColumns}
          dataSource={features?.string_matches}
          rowKey="type"
          pagination={false}
        />
      </Card>

      <Card title="YARA规则匹配" style={{ marginBottom: 16 }}>
        <Table
          columns={yaraColumns}
          dataSource={features?.yara_matches}
          rowKey="name"
          pagination={false}
        />
      </Card>

      <Card title="操作码特征" style={{ marginBottom: 16 }}>
        <Table
          columns={opcodeColumns}
          dataSource={features?.opcode_features}
          rowKey="opcode"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
          }}
        />
      </Card>

      <Card title="布尔特征">
        <Descriptions bordered>
          {Object.entries(features?.boolean_features || {}).map(([key, value]) => (
            <Descriptions.Item key={key} label={key}>
              <Tag color={value ? 'red' : 'green'}>
                {value ? '是' : '否'}
              </Tag>
            </Descriptions.Item>
          ))}
        </Descriptions>
      </Card>
    </div>
  );
};

export default FeatureEngineering;
