import React from 'react';
import { PageContainer } from '@ant-design/pro-layout';
import { Card, Row, Col, Statistic, Table, Descriptions } from 'antd';
import { useParams } from 'umi';
import { fetchSampleDetail } from '@/services/api';
import ReactECharts from 'echarts-for-react';
import { useRequest } from 'ahooks';

const SampleDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const { data: sample, loading } = useRequest(() => fetchSampleDetail(Number(id)));

  if (loading || !sample) {
    return <div>Loading...</div>;
  }

  // 节区特征图表配置
  const sectionChartOption = {
    title: {
      text: '节区特征分布',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    legend: {
      data: ['大小', '熵值'],
      top: 30
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: ['可读', '可写', '可执行']
    },
    yAxis: [
      {
        type: 'value',
        name: '大小',
        position: 'left'
      },
      {
        type: 'value',
        name: '熵值',
        position: 'right'
      }
    ],
    series: [
      {
        name: '大小',
        type: 'bar',
        data: [
          sample.engineered_features.section_features.size_R,
          sample.engineered_features.section_features.size_W,
          sample.engineered_features.section_features.size_X
        ]
      },
      {
        name: '熵值',
        type: 'line',
        yAxisIndex: 1,
        data: [
          sample.engineered_features.section_features.entr_R,
          sample.engineered_features.section_features.entr_W,
          sample.engineered_features.section_features.entr_X
        ]
      }
    ]
  };

  // 节区权重图表配置
  const sectionWeightChartOption = {
    title: {
      text: '节区权重分布',
      left: 'center'
    },
    tooltip: {
      trigger: 'item'
    },
    legend: {
      orient: 'vertical',
      left: 'left'
    },
    series: [
      {
        name: '节区权重',
        type: 'pie',
        radius: '50%',
        data: [
          { value: sample.engineered_features.section_features.size_R_weight, name: '可读权重' },
          { value: sample.engineered_features.section_features.size_W_weight, name: '可写权重' },
          { value: sample.engineered_features.section_features.size_X_weight, name: '可执行权重' }
        ],
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  };

  // 字符串匹配特征图表配置
  const stringMatchChartOption = {
    title: {
      text: '字符串匹配特征',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    legend: {
      data: ['数量', '平均长度'],
      top: 30
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: ['MZ', 'PE', '矿池', 'CPU', 'GPU', '数字货币']
    },
    yAxis: [
      {
        type: 'value',
        name: '数量',
        position: 'left'
      },
      {
        type: 'value',
        name: '平均长度',
        position: 'right'
      }
    ],
    series: [
      {
        name: '数量',
        type: 'bar',
        data: [
          sample.engineered_features.string_match.mz_count,
          sample.engineered_features.string_match.pe_count,
          sample.engineered_features.string_match.pool_count,
          sample.engineered_features.string_match.cpu_count,
          sample.engineered_features.string_match.gpu_count,
          sample.engineered_features.string_match.coin_count
        ]
      },
      {
        name: '平均长度',
        type: 'line',
        yAxisIndex: 1,
        data: [
          sample.engineered_features.string_match.mz_mean,
          sample.engineered_features.string_match.pe_mean,
          sample.engineered_features.string_match.pool_mean,
          sample.engineered_features.string_match.cpu_mean,
          sample.engineered_features.string_match.gpu_mean,
          sample.engineered_features.string_match.coin_mean
        ]
      }
    ]
  };

  return (
    <PageContainer>
      <Card title="样本基本信息">
        <Descriptions>
          <Descriptions.Item label="文件名称">{sample.file_name}</Descriptions.Item>
          <Descriptions.Item label="文件大小">{sample.file_size} bytes</Descriptions.Item>
          <Descriptions.Item label="文件哈希">{sample.file_hash}</Descriptions.Item>
          <Descriptions.Item label="分析结果">
            {sample.classification_result}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={12}>
          <Card title="节区特征分析">
            <ReactECharts option={sectionChartOption} style={{ height: 400 }} />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="节区权重分析">
            <ReactECharts option={sectionWeightChartOption} style={{ height: 400 }} />
          </Card>
        </Col>
      </Row>

      <Card title="字符串匹配特征分析" style={{ marginTop: 16 }}>
        <ReactECharts option={stringMatchChartOption} style={{ height: 400 }} />
      </Card>

      <Card title="详细特征数据" style={{ marginTop: 16 }}>
        <Descriptions title="节区特征" column={3}>
          <Descriptions.Item label="入口节区">{sample.engineered_features.section_features.entry}</Descriptions.Item>
          <Descriptions.Item label="节区数量">{sample.engineered_features.section_features.section_num}</Descriptions.Item>
          <Descriptions.Item label="资源节区数量">{sample.engineered_features.section_features.rsrc_num}</Descriptions.Item>
          <Descriptions.Item label="可读节区大小">{sample.engineered_features.section_features.size_R}</Descriptions.Item>
          <Descriptions.Item label="可写节区大小">{sample.engineered_features.section_features.size_W}</Descriptions.Item>
          <Descriptions.Item label="可执行节区大小">{sample.engineered_features.section_features.size_X}</Descriptions.Item>
          <Descriptions.Item label="可读节区熵值">{sample.engineered_features.section_features.entr_R}</Descriptions.Item>
          <Descriptions.Item label="可写节区熵值">{sample.engineered_features.section_features.entr_W}</Descriptions.Item>
          <Descriptions.Item label="可执行节区熵值">{sample.engineered_features.section_features.entr_X}</Descriptions.Item>
        </Descriptions>

        <Descriptions title="字符串匹配特征" column={3} style={{ marginTop: 16 }}>
          <Descriptions.Item label="MZ标记数量">{sample.engineered_features.string_match.mz_count}</Descriptions.Item>
          <Descriptions.Item label="MZ平均长度">{sample.engineered_features.string_match.mz_mean}</Descriptions.Item>
          <Descriptions.Item label="PE标记数量">{sample.engineered_features.string_match.pe_count}</Descriptions.Item>
          <Descriptions.Item label="PE平均长度">{sample.engineered_features.string_match.pe_mean}</Descriptions.Item>
          <Descriptions.Item label="矿池关键词数量">{sample.engineered_features.string_match.pool_count}</Descriptions.Item>
          <Descriptions.Item label="矿池平均长度">{sample.engineered_features.string_match.pool_mean}</Descriptions.Item>
          <Descriptions.Item label="CPU关键词数量">{sample.engineered_features.string_match.cpu_count}</Descriptions.Item>
          <Descriptions.Item label="CPU平均长度">{sample.engineered_features.string_match.cpu_mean}</Descriptions.Item>
          <Descriptions.Item label="GPU关键词数量">{sample.engineered_features.string_match.gpu_count}</Descriptions.Item>
          <Descriptions.Item label="GPU平均长度">{sample.engineered_features.string_match.gpu_mean}</Descriptions.Item>
          <Descriptions.Item label="数字货币关键词数量">{sample.engineered_features.string_match.coin_count}</Descriptions.Item>
          <Descriptions.Item label="数字货币平均长度">{sample.engineered_features.string_match.coin_mean}</Descriptions.Item>
        </Descriptions>
      </Card>
    </PageContainer>
  );
};

export default SampleDetail; 