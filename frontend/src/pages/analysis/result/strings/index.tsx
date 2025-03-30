import { Card, Table, Input, Spin, Tabs } from 'antd';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";

const { TabPane } = Tabs;
const { Search } = Input;

const StringAnalysis: FC = () => {
  const { currentFileId } = useAnalysisModel();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any>(null);
  const [searchText, setSearchText] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get(`http://localhost:5000/api/v1/analysis/result/strings/${currentFileId}`);
        setData(data);
      } catch (error) {
        console.error('获取字符串分析结果失败', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [currentFileId]);

  if (loading) {
    return <Spin />;
  }

  const stringInfo = data?.string_info || {};

  // 将字符串数据解析为数组
  const parseStringData = (str: string) => {
    try {
      return JSON.parse(str);
    } catch (e) {
      return [];
    }
  };

  const columns = [
    {
      title: '偏移量',
      dataIndex: 'offset',
      key: 'offset',
      width: 120,
      render: (offset: number) => `0x${offset.toString(16).toUpperCase()}`,
    },
    {
      title: '字符串内容',
      dataIndex: 'string',
      key: 'string',
      ellipsis: true,
      filteredValue: [searchText],
      onFilter: (value: string, record) =>
        record.string.toLowerCase().includes(value.toLowerCase()),
    },
  ];

  return (
    <Card
      title="字符串信息"
      extra={
        <Search
          placeholder="搜索字符串"
          onSearch={setSearchText}
          style={{ width: 200 }}
        />
      }
    >
      <Tabs defaultActiveKey="1">
        <TabPane tab="ASCII字符串" key="1">
          <Table
            columns={columns}
            dataSource={parseStringData(stringInfo.ascii_strings)}
            rowKey="offset"
            scroll={{ y: 300 }}
            pagination={false}
          />
        </TabPane>

        <TabPane tab="Unicode字符串" key="2">
          <Table
            columns={columns}
            dataSource={parseStringData(stringInfo.unicode_strings)}
            rowKey="offset"
            scroll={{ y: 300 }}
            pagination={false}
          />
        </TabPane>
      </Tabs>
    </Card>
  );
};

export default StringAnalysis;
