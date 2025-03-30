import { Card, Table, Tag, Spin } from 'antd';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";

const SigmaMatches: FC = () => {
  const { currentFileId } = useAnalysisModel();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get(`http://localhost:5000/api/v1/analysis/result/sigma-rules/${currentFileId}`);
        setData(data);
      } catch (error) {
        console.error('获取Sigma规则匹配结果失败', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [currentFileId]);

  if (loading) {
    return <Spin />;
  }

  const sigmaMatches = data?.sigma_matches || [];

  console.log(sigmaMatches)

  const columns = [
    {
      title: '规则ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: '规则名称',
      dataIndex: 'title',
      key: 'title',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'true' ? 'red' : 'green'}>
          {status === 'true' ? '匹配' : '不匹配'}
        </Tag>
      ),
    },
    {
      title: '详细信息',
      dataIndex: 'details',
      key: 'details',
      render: (details: any) => (
        <pre style={{ maxHeight: '200px', overflow: 'auto' }}>
          {JSON.stringify(details, null, 2)}
        </pre>
      ),
    },
  ];

  return (
    <Card title="Sigma规则匹配结果">
      <Table
        columns={columns}
        dataSource={sigmaMatches}
        rowKey="id"
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showQuickJumper: true,
        }}
      />
    </Card>
  );
};

export default SigmaMatches;
