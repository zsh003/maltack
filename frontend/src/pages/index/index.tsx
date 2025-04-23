import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Typography, Button } from 'antd';
import { FileProtectOutlined, BugOutlined, SafetyOutlined, FileSearchOutlined } from '@ant-design/icons';
import { history } from 'umi';
import { fetchStats } from '@/services/api';
import ReactECharts from 'echarts-for-react';

const { Title, Paragraph } = Typography;

const HomePage: React.FC = () => {
  const [stats, setStats] = useState({
    total_samples: 0,
    malicious_samples: 0,
    benign_samples: 0,
    detection_rate: 0,
  });

  useEffect(() => {
    const loadStats = async () => {
      try {
        const data = await fetchStats();
        setStats(data);
      } catch (error) {
        console.error('加载统计数据失败', error);
      }
    };

    loadStats();
  }, []);

  const pieOption = {
    title: {
      text: '样本分布',
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

  return (
    <div className="home-page" style={{ padding: '24px' }}>
      <Title>PE恶意软件特征分析平台</Title>
      <Paragraph>
        本平台提供基于集成学习的恶意PE软件特征检测与识别服务，帮助您分析可疑软件的特征，并通过多种特征维度判断样本是否为恶意软件。
      </Paragraph>

      <div className="action-buttons" style={{ marginBottom: '24px' }}>
        <Button type="primary" size="large" onClick={() => history.push('/upload')}>
          上传新样本
        </Button>
        <Button style={{ marginLeft: '16px' }} size="large" onClick={() => history.push('/samples')}>
          查看所有样本
        </Button>
        <Button style={{ marginLeft: '16px' }} size="large" onClick={() => history.push('/dashboard')}>
          仪表盘
        </Button>
        <Button style={{ marginLeft: '16px' }} size="large" onClick={() => history.push('/model')}>
          机器学习分析
        </Button>
      </div>

      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card>
            <ReactECharts option={pieOption} style={{ height: '300px' }} />
          </Card>
        </Col>
        <Col span={12}>
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Card>
                <Statistic
                  title="总样本数"
                  value={stats.total_samples}
                  prefix={<FileSearchOutlined />}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card>
                <Statistic
                  title="恶意样本"
                  value={stats.malicious_samples}
                  prefix={<BugOutlined />}
                  valueStyle={{ color: '#f5222d' }}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card>
                <Statistic
                  title="正常样本"
                  value={stats.benign_samples}
                  prefix={<SafetyOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col span={12}>
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
        </Col>
      </Row>

      <Card style={{ marginTop: '24px' }}>
        <Title level={4}>平台能力介绍</Title>
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <Card title="直方图特征分析" className="feature-card">
              <ul>
                <li>字节分布直方图 (256维)</li>
                <li>字节熵直方图 (256维)</li>
                <li>检测加密/填充数据</li>
                <li>识别代码段/混淆区域</li>
              </ul>
            </Card>
          </Col>
          <Col span={8}>
            <Card title="PE静态特征分析" className="feature-card">
              <ul>
                <li>PE文件头信息分析</li>
                <li>节区特征提取与分析</li>
                <li>导入/导出表分析</li>
                <li>识别异常PE结构</li>
              </ul>
            </Card>
          </Col>
          <Col span={8}>
            <Card title="特征工程分析" className="feature-card">
              <ul>
                <li>Yara规则匹配</li>
                <li>恶意字符串检测</li>
                <li>Opcode特征分析</li>
                <li>多维度综合评估</li>
              </ul>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default HomePage; 