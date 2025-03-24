import { useRequest } from '@umijs/max';
import { Card, Table, Tag, Spin } from 'antd';
import type { FC } from 'react';
import { getYaraRules } from '../service';

const YaraMatches: FC = () => {
  const { data, loading } = useRequest(() => getYaraRules());

  if (loading) {
    return <Spin />;
  }

  const yaraMatches = data?.yara_matches || [];

  const columns = [
    {
      title: '规则名称',
      dataIndex: 'rule_name',
      key: 'rule_name',
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <>
          {tags.map((tag) => (
            <Tag key={tag} color="blue">
              {tag}
            </Tag>
          ))}
        </>
      ),
    },
    {
      title: '匹配字符串',
      dataIndex: 'strings',
      key: 'strings',
      render: (strings: any[]) => (
        <Table
          size="small"
          dataSource={strings}
          columns={[
            {
              title: '标识符',
              dataIndex: 'identifier',
              key: 'identifier',
            },
            {
              title: '数据',
              dataIndex: 'data',
              key: 'data',
            },
            {
              title: '偏移量',
              dataIndex: 'offset',
              key: 'offset',
            },
          ]}
        />
      ),
    },
  ];

  return (
    <Card title="Yara规则匹配结果">
      <Table
        columns={columns}
        dataSource={yaraMatches}
        rowKey="rule_name"
        expandable={{
          expandedRowRender: (record) => (
            <div>
              <h4>元数据：</h4>
              <pre>{JSON.stringify(record.meta, null, 2)}</pre>
            </div>
          ),
        }}
      />
    </Card>
  );
};

export default YaraMatches;
