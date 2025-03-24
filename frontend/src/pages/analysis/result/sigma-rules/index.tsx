import { useRequest } from '@umijs/max';
import { Card, Table, Tag, Spin } from 'antd';
import type { FC } from 'react';
import { getSigmaRules } from '../service';

const SigmaMatches: FC = () => {
  const { data, loading } = useRequest(() => getSigmaRules());

  if (loading) {
    return <Spin />;
  }

  const sigmaMatches = data?.sigma_matches || [];

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
