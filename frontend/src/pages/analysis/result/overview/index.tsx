import { useRequest } from '@umijs/max';
import { Card, Descriptions, Table, Tag, Spin, Tabs } from 'antd';
import type { FC } from 'react';
import { getAnalysisResult } from '../service';

const { TabPane } = Tabs;

const Overview: FC = () => {

  const { data, loading } = useRequest(() => getAnalysisResult());

  if (loading) {
    return <Spin />;
  }

  const { basic_info, pe_info, yara_matches, sigma_matches, string_info } = data || {};

  const yaraColumns = [
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
  ];

  const sigmaColumns = [
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
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'true' ? 'red' : 'green'}>
          {status === 'true' ? '匹配' : '不匹配'}
        </Tag>
      ),
    },
  ];

  const stringColumns = [
    {
      title: '偏移量',
      dataIndex: 'offset',
      key: 'offset',
      width: 120,
      render: (offset: number) => `0x${offset.toString(16)}`,
    },
    {
      title: '字符串内容',
      dataIndex: 'string',
      key: 'string',
      ellipsis: true,
    },
  ];

  return (
    <div>
      <Card title="基本信息" style={{ marginBottom: 16 }}>
        <Descriptions column={2}>
          <Descriptions.Item label="文件名">{basic_info?.file_name}</Descriptions.Item>
          <Descriptions.Item label="文件大小">{basic_info?.file_size} 字节</Descriptions.Item>
          <Descriptions.Item label="文件类型">{basic_info?.file_type}</Descriptions.Item>
          <Descriptions.Item label="MIME类型">{basic_info?.mime_type}</Descriptions.Item>
          <Descriptions.Item label="MD5">{basic_info?.md5}</Descriptions.Item>
          <Descriptions.Item label="SHA1">{basic_info?.sha1}</Descriptions.Item>
          <Descriptions.Item label="SHA256">{basic_info?.sha256}</Descriptions.Item>
          <Descriptions.Item label="分析时间">{basic_info?.analyze_time}</Descriptions.Item>
        </Descriptions>
      </Card>

      {pe_info && (
        <Card title="PE文件信息" style={{ marginBottom: 16 }}>
          <Descriptions column={2} bordered>
            <Descriptions.Item label="机器类型">{pe_info.machine_type}</Descriptions.Item>
            <Descriptions.Item label="时间戳">{pe_info.timestamp}</Descriptions.Item>
            <Descriptions.Item label="子系统">{pe_info.subsystem}</Descriptions.Item>
            <Descriptions.Item label="DLL特征">{pe_info.dll_characteristics}</Descriptions.Item>
          </Descriptions>
        </Card>
      )}

      <Card title="规则匹配结果">
        <Tabs defaultActiveKey="yara">
          <TabPane tab="Yara规则匹配" key="yara">
            <Table
              columns={yaraColumns}
              dataSource={yara_matches || []}
              rowKey="rule_name"
              pagination={false}
            />
          </TabPane>
          <TabPane tab="Sigma规则匹配" key="sigma">
            <Table
              columns={sigmaColumns}
              dataSource={sigma_matches || []}
              rowKey="id"
              pagination={false}
            />
          </TabPane>
        </Tabs>
      </Card>

      <Card title="字符串分析" style={{ marginTop: 16 }}>
        <Tabs defaultActiveKey="ascii">
          <TabPane tab="ASCII字符串" key="ascii">
            <Table
              columns={stringColumns}
              dataSource={string_info?.ascii_strings || []}
              rowKey="offset"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
              }}
            />
          </TabPane>
          <TabPane tab="Unicode字符串" key="unicode">
            <Table
              columns={stringColumns}
              dataSource={string_info?.unicode_strings || []}
              rowKey="offset"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
              }}
            />
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default Overview;
