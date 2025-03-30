import { Card, Table, Tag, Spin } from 'antd';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";

const YaraMatches: FC = () => {
  const { currentFileId } = useAnalysisModel();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get(`http://localhost:5000/api/v1/analysis/result/yara-rules/${currentFileId}`);
        setData(data);
      } catch (error) {
        console.error('获取YARA规则匹配结果失败', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [currentFileId]);

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
