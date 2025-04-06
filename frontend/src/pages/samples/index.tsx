import React, { useEffect, useState } from 'react';
import { Table, Card, Typography, Tag, Button, Input, Space, Tooltip } from 'antd';
import { SearchOutlined, FileSearchOutlined, UploadOutlined } from '@ant-design/icons';
import { fetchSamples } from '@/services/api';
import { history } from 'umi';
import Highlighter from 'react-highlight-words';

const { Title } = Typography;
const { Search } = Input;

const SamplesPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [samples, setSamples] = useState<any[]>([]);
  const [searchText, setSearchText] = useState('');
  const [searchedColumn, setSearchedColumn] = useState('');
  const [filteredSamples, setFilteredSamples] = useState<any[]>([]);

  useEffect(() => {
    const loadSamples = async () => {
      setLoading(true);
      try {
        const data = await fetchSamples();
        setSamples(data);
        setFilteredSamples(data);
      } catch (error) {
        console.error('加载样本列表失败', error);
      } finally {
        setLoading(false);
      }
    };

    loadSamples();
  }, []);

  const handleSearch = (value: string) => {
    setSearchText(value);
    if (!value) {
      setFilteredSamples(samples);
      return;
    }

    const filtered = samples.filter(sample => 
      sample.file_name.toLowerCase().includes(value.toLowerCase()) ||
      sample.file_hash.toLowerCase().includes(value.toLowerCase())
    );
    setFilteredSamples(filtered);
  };

  const getColumnSearchProps = (dataIndex: string) => ({
    filterDropdown: ({ setSelectedKeys, selectedKeys, confirm, clearFilters }: any) => (
      <div style={{ padding: 8 }}>
        <Input
          placeholder={`搜索 ${dataIndex}`}
          value={selectedKeys[0]}
          onChange={e => setSelectedKeys(e.target.value ? [e.target.value] : [])}
          onPressEnter={() => handleColumnSearch(selectedKeys, confirm, dataIndex)}
          style={{ width: 188, marginBottom: 8, display: 'block' }}
        />
        <Space>
          <Button
            type="primary"
            onClick={() => handleColumnSearch(selectedKeys, confirm, dataIndex)}
            icon={<SearchOutlined />}
            size="small"
            style={{ width: 90 }}
          >
            搜索
          </Button>
          <Button onClick={() => handleColumnReset(clearFilters)} size="small" style={{ width: 90 }}>
            重置
          </Button>
        </Space>
      </div>
    ),
    filterIcon: (filtered: boolean) => <SearchOutlined style={{ color: filtered ? '#1890ff' : undefined }} />,
    onFilter: (value: string, record: any) =>
      record[dataIndex]
        ? record[dataIndex].toString().toLowerCase().includes(value.toLowerCase())
        : false,
    render: (text: string) =>
      searchedColumn === dataIndex ? (
        <Highlighter
          highlightStyle={{ backgroundColor: '#ffc069', padding: 0 }}
          searchWords={[searchText]}
          autoEscape
          textToHighlight={text ? text.toString() : ''}
        />
      ) : (
        text
      ),
  });

  const handleColumnSearch = (selectedKeys: string[], confirm: () => void, dataIndex: string) => {
    confirm();
    setSearchedColumn(dataIndex);
    setSearchText(selectedKeys[0]);
  };

  const handleColumnReset = (clearFilters: () => void) => {
    clearFilters();
    setSearchText('');
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: '文件名',
      dataIndex: 'file_name',
      key: 'file_name',
      ...getColumnSearchProps('file_name'),
      render: (text: string, record: any) => (
        <a onClick={() => history.push(`/samples/${record.id}`)}>{text}</a>
      ),
    },
    {
      title: '文件哈希',
      dataIndex: 'file_hash',
      key: 'file_hash',
      ...getColumnSearchProps('file_hash'),
      render: (hash: string) => (
        <Tooltip title={hash}>
          {hash.substring(0, 8)}...{hash.substring(hash.length - 8)}
        </Tooltip>
      ),
    },
    {
      title: '文件大小',
      dataIndex: 'file_size',
      key: 'file_size',
      sorter: (a: any, b: any) => a.file_size - b.file_size,
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`,
    },
    {
      title: '分析时间',
      dataIndex: 'analysis_time',
      key: 'analysis_time',
      sorter: (a: any, b: any) => new Date(a.analysis_time).getTime() - new Date(b.analysis_time).getTime(),
    },
    {
      title: '检测结果',
      dataIndex: 'is_malicious',
      key: 'is_malicious',
      filters: [
        { text: '恶意软件', value: 1 },
        { text: '正常软件', value: 0 },
      ],
      onFilter: (value: number, record: any) => record.is_malicious === value,
      render: (isMalicious: number) => (
        <Tag color={isMalicious === 1 ? 'error' : 'success'}>
          {isMalicious === 1 ? '恶意软件' : '正常软件'}
        </Tag>
      ),
    },
  ];

  return (
    <div className="samples-page" style={{ padding: '24px' }}>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <Title level={2}>样本列表</Title>
        <Space>
          <Search 
            placeholder="搜索文件名或哈希值" 
            onSearch={handleSearch} 
            style={{ width: 300 }} 
            enterButton 
          />
          <Button 
            type="primary" 
            icon={<UploadOutlined />}
            onClick={() => history.push('/upload')}
          >
            上传样本
          </Button>
        </Space>
      </div>

      <Card>
        <Table 
          columns={columns} 
          dataSource={filteredSamples} 
          rowKey="id" 
          loading={loading}
          pagination={{ 
            pageSize: 10, 
            showSizeChanger: true, 
            showTotal: (total) => `共 ${total} 个样本` 
          }}
          onRow={(record) => ({
            onClick: () => {
              history.push(`/samples/${record.id}`);
            },
          })}
        />
      </Card>
    </div>
  );
};

export default SamplesPage; 