import { useRequest } from '@umijs/max';
import { Card, Table, Tabs, Spin, Input } from 'antd';
import type { FC } from 'react';
import { useState } from 'react';
import { getStrings } from '../service';

const { Search } = Input;
const { TabPane } = Tabs;

const StringAnalysis: FC = () => {
  const { data, loading } = useRequest(() => getStrings());
  const [searchText, setSearchText] = useState('');

  if (loading) {
    return <Spin />;
  }

  const stringInfo = data?.string_info || {};
  const { ascii_strings = [], unicode_strings = [] } = stringInfo;

  const filterStrings = (strings: any[]) => {
    if (!searchText) return strings;
    return strings.filter((item) =>
      item.string.toLowerCase().includes(searchText.toLowerCase()),
    );
  };

  const columns = [
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
    <Card title="字符串分析">
      <Search
        placeholder="搜索字符串"
        allowClear
        onChange={(e) => setSearchText(e.target.value)}
        style={{ marginBottom: 16 }}
      />
      <Tabs defaultActiveKey="ascii">
        <TabPane tab="ASCII字符串" key="ascii">
          <Table
            columns={columns}
            dataSource={filterStrings(ascii_strings)}
            rowKey="offset"
            pagination={{
              pageSize: 20,
              showSizeChanger: true,
              showQuickJumper: true,
            }}
            scroll={{ y: 400 }}
          />
        </TabPane>
        <TabPane tab="Unicode字符串" key="unicode">
          <Table
            columns={columns}
            dataSource={filterStrings(unicode_strings)}
            rowKey="offset"
            pagination={{
              pageSize: 20,
              showSizeChanger: true,
              showQuickJumper: true,
            }}
            scroll={{ y: 400 }}
          />
        </TabPane>
      </Tabs>
    </Card>
  );
};

export default StringAnalysis;
