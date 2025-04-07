import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Typography, Table, Tag, Spin } from 'antd';
import { FileProtectOutlined, BugOutlined, SafetyOutlined, FileSearchOutlined } from '@ant-design/icons';
import { fetchSamples, fetchStats } from '@/services/api';
import ReactECharts from 'echarts-for-react';
import { history } from 'umi';

const { Title } = Typography;

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total_samples: 0,
    malicious_samples: 0,
    benign_samples: 0,
    detection_rate: 0,
  });
  const [samples, setSamples] = useState<any[]>([]);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [statsData, samplesData] = await Promise.all([
          fetchStats(),
          fetchSamples(),
        ]);
        
        setStats(statsData);
        setSamples(samplesData);
      } catch (error) {
        console.error('加载数据失败', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // 样本分布饼图
  const pieOption = {
    title: {
      text: '样本分类统计',
      left: 'center',
    },
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)',
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      data: ['恶意样本', '正常样本'],
    },
    series: [
      {
        name: '样本类型',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        label: {
          show: false,
          position: 'center',
        },
        emphasis: {
          label: {
            show: true,
            fontSize: '18',
            fontWeight: 'bold',
          },
        },
        labelLine: {
          show: false,
        },
        data: [
          { value: stats.malicious_samples, name: '恶意样本', itemStyle: { color: '#f5222d' } },
          { value: stats.benign_samples, name: '正常样本', itemStyle: { color: '#52c41a' } },
        ],
      },
    ],
  };

  // 样本大小分布图
  const getSampleSizeData = () => {
    const sizeRanges = [
      { min: 0, max: 100 * 1024, label: '< 100KB' },
      { min: 100 * 1024, max: 500 * 1024, label: '100KB-500KB' },
      { min: 500 * 1024, max: 1000 * 1024, label: '500KB-1MB' },
      { min: 1000 * 1024, max: 5000 * 1024, label: '1MB-5MB' },
      { min: 5000 * 1024, max: Infinity, label: '> 5MB' },
    ];

    const counts = Array(sizeRanges.length).fill(0);
    
    samples.forEach(sample => {
      const size = sample.file_size;
      const rangeIndex = sizeRanges.findIndex(range => size >= range.min && size < range.max);
      if (rangeIndex !== -1) {
        counts[rangeIndex]++;
      }
    });

    return {
      labels: sizeRanges.map(r => r.label),
      counts,
    };
  };

  const sizeData = getSampleSizeData();
  const barOption = {
    title: {
      text: '样本大小分布',
      left: 'center',
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
    },
    xAxis: {
      type: 'category',
      data: sizeData.labels,
      axisTick: {
        alignWithLabel: true,
      },
    },
    yAxis: {
      type: 'value',
      name: '样本数量',
    },
    series: [
      {
        name: '样本数量',
        type: 'bar',
        barWidth: '60%',
        data: sizeData.counts,
        itemStyle: {
          color: '#1890ff',
        },
      },
    ],
  };

  // 近期样本表格
  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: '文件名',
      dataIndex: 'file_name',
      key: 'file_name',
      render: (text: string) => <a>{text}</a>,
    },
    {
      title: '文件大小',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`,
    },
    {
      title: '分析时间',
      dataIndex: 'analysis_time',
      key: 'analysis_time',
    },
    {
      title: '检测结果',
      dataIndex: 'is_malicious',
      key: 'is_malicious',
      render: (isMalicious: number) => (
        <Tag color={isMalicious === 1 ? 'error' : 'success'}>
          {isMalicious === 1 ? '恶意软件' : '正常软件'}
        </Tag>
      ),
    },
  ];

  return (
    <div className="dashboard-page" style={{ padding: '24px' }}>
      <Title level={2}>数据仪表盘</Title>
      
      <Spin spinning={loading}>
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总样本数"
                value={stats.total_samples}
                prefix={<FileSearchOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="恶意样本"
                value={stats.malicious_samples}
                prefix={<BugOutlined />}
                valueStyle={{ color: '#f5222d' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="正常样本"
                value={stats.benign_samples}
                prefix={<SafetyOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="检出率"
                value={stats.detection_rate}
                suffix="%"
                prefix={<FileProtectOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
          <Col span={12}>
            <Card title="样本分类统计">
              <ReactECharts option={pieOption} style={{ height: '300px' }} />
            </Card>
          </Col>
          <Col span={12}>
            <Card title="样本大小分布">
              <ReactECharts option={barOption} style={{ height: '300px' }} />
            </Card>
          </Col>
        </Row>

        <Card title="最近分析的样本" style={{ marginTop: '16px' }}>
          <Table 
            columns={columns} 
            dataSource={samples.slice(0, 5)}
            rowKey="id" 
            onRow={(record) => ({
              onClick: () => {
                history.push(`/samples/${record.id}`);
              },
            })}
          />
        </Card>
      </Spin>
    </div>
  );
};

export default Dashboard; 