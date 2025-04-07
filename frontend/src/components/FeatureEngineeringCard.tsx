import React from 'react';
import { Card, Collapse, Descriptions, Progress, Statistic, Row, Col, Typography, Tag } from 'antd';
import { WarningOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';

const { Panel } = Collapse;
const { Title } = Typography;

interface FeatureEngineeringCardProps {
  sectionFeatures: any;
  stringMatch: any;
  yaraMatch: any;
  stringCount: any;
  opcodeFeatures: any;
}

const FeatureEngineeringCard: React.FC<FeatureEngineeringCardProps> = ({
  sectionFeatures,
  stringMatch,
  yaraMatch,
  stringCount,
  opcodeFeatures,
}) => {
  
  // 准备雷达图数据
  const radarOption = {
    title: {
      text: '特征分布雷达图',
    },
    tooltip: {},
    radar: {
      // shape: 'circle',
      indicator: [
        { name: '节区信息', max: 10 },
        { name: '字符匹配', max: 10 },
        { name: 'Yara匹配', max: 10 },
        { name: '关键字统计', max: 10 },
        { name: 'Opcode分析', max: 10 },
      ]
    },
    series: [{
      name: '特征评分',
      type: 'radar',
      data: [
        {
          value: [
            // 节区信息评分 - 根据可疑性评分
            Math.min(10, (
              (sectionFeatures.entr_X > 7 ? 4 : 0) + 
              (sectionFeatures.size_W / sectionFeatures.file_size > 0.5 ? 3 : 0) +
              (sectionFeatures.rsrc_num > 1 ? 3 : 0)
            )),
            // 字符匹配评分
            Math.min(10, (
              stringMatch.btc_count + stringMatch.ltc_count + stringMatch.xmr_count +
              (stringMatch.urls_count > 2 ? 3 : 0) +
              (stringMatch.ips_count > 2 ? 3 : 0)
            )),
            // Yara匹配评分
            Math.min(10, (yaraMatch.packer_count * 5 + yaraMatch.yargen_count * 2)),
            // 关键字统计评分
            Math.min(10, (
              (stringCount.av_count > 5 ? 3 : 0) +
              (stringCount.pool_name_count > 10 ? 5 : stringCount.pool_name_count > 5 ? 3 : 0) +
              (stringCount.coin_name_count > 3 ? 4 : 0)
            )),
            // Opcode分析评分
            Math.min(10, (
              (opcodeFeatures.opcode_uniq > 150 ? 4 : 0) +
              (opcodeFeatures.opcode_var > 5000 ? 3 : 0) +
              (opcodeFeatures.opcode_count > 300 ? 3 : 0)
            )),
          ],
          name: '特征得分',
          areaStyle: {
            color: 'rgba(255, 87, 51, 0.3)'
          },
          lineStyle: {
            color: 'rgb(255, 87, 51)'
          }
        }
      ]
    }]
  };

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
          sectionFeatures.size_R,
          sectionFeatures.size_W,
          sectionFeatures.size_X
        ]
      },
      {
        name: '熵值',
        type: 'line',
        yAxisIndex: 1,
        data: [
          sectionFeatures.entr_R,
          sectionFeatures.entr_W,
          sectionFeatures.entr_X
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
          { value: sectionFeatures.size_R_weight, name: '可读权重' },
          { value: sectionFeatures.size_W_weight, name: '可写权重' },
          { value: sectionFeatures.size_X_weight, name: '可执行权重' }
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
          stringMatch.mz_count,
          stringMatch.pe_count,
          stringMatch.pool_count,
          stringMatch.cpu_count,
          stringMatch.gpu_count,
          stringMatch.coin_count
        ]
      },
      {
        name: '平均长度',
        type: 'line',
        yAxisIndex: 1,
        data: [
          stringMatch.mz_mean,
          stringMatch.pe_mean,
          stringMatch.pool_mean,
          stringMatch.cpu_mean,
          stringMatch.gpu_mean,
          stringMatch.coin_mean
        ]
      }
    ]
  };

  return (
    <Card title="特征工程分析" className="feature-engineering-card">
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <ReactECharts option={radarOption} style={{ height: '350px' }} />
        </Col>
        <Col span={12}>
          <Card title="关键指标" className="metrics-card">
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic 
                  title="可执行节区熵值" 
                  value={sectionFeatures.entr_X.toFixed(2)} 
                  suffix="/8.0"
                  valueStyle={{ color: sectionFeatures.entr_X > 7 ? '#cf1322' : '#3f8600' }}
                />
                <Progress 
                  percent={sectionFeatures.entr_X / 8 * 100} 
                  status={sectionFeatures.entr_X > 7 ? 'exception' : 'normal'} 
                  strokeColor={sectionFeatures.entr_X > 7 ? '#cf1322' : '#3f8600'}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="矿池关键词数" 
                  value={stringCount.pool_name_count} 
                  valueStyle={{ color: stringCount.pool_name_count > 5 ? '#cf1322' : '#3f8600' }}
                />
                <Progress 
                  percent={Math.min(100, stringCount.pool_name_count * 5)} 
                  status={stringCount.pool_name_count > 5 ? 'exception' : 'normal'} 
                  strokeColor={stringCount.pool_name_count > 5 ? '#cf1322' : '#3f8600'}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="壳检测次数" 
                  value={yaraMatch.packer_count} 
                  valueStyle={{ color: yaraMatch.packer_count > 0 ? '#cf1322' : '#3f8600' }}
                />
                <Tag color={yaraMatch.packer_count > 0 ? 'error' : 'success'}>
                  {yaraMatch.packer_count > 0 ? '检测到壳' : '无壳'}
                </Tag>
              </Col>
              <Col span={12}>
                <Statistic 
                  title="反杀软检测" 
                  value={stringCount.av_count} 
                  suffix="次"
                  valueStyle={{ color: stringCount.av_count > 3 ? '#cf1322' : '#3f8600' }}
                />
                <Tag color={stringCount.av_count > 3 ? 'error' : 'success'}>
                  {stringCount.av_count > 3 ? '存在反杀软行为' : '无反杀软行为'}
                </Tag>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      <Collapse ghost style={{ marginTop: 16 }}>
        <Panel header="节区特征详情" key="1">
          <Descriptions bordered size="small" column={3}>
            <Descriptions.Item label="入口点长度">{sectionFeatures.entry}</Descriptions.Item>
            <Descriptions.Item label="可读节区大小">{(sectionFeatures.size_R / 1024).toFixed(2)} KB</Descriptions.Item>
            <Descriptions.Item label="可写节区大小">{(sectionFeatures.size_W / 1024).toFixed(2)} KB</Descriptions.Item>
            <Descriptions.Item label="可执行节区大小">{(sectionFeatures.size_X / 1024).toFixed(2)} KB</Descriptions.Item>
            <Descriptions.Item label="可读节区熵值">{sectionFeatures.entr_R.toFixed(2)}</Descriptions.Item>
            <Descriptions.Item label="可写节区熵值">{sectionFeatures.entr_W.toFixed(2)}</Descriptions.Item>
            <Descriptions.Item label="可执行节区熵值">{sectionFeatures.entr_X.toFixed(2)}</Descriptions.Item>
            <Descriptions.Item label="资源节区数量">{sectionFeatures.rsrc_num}</Descriptions.Item>
            <Descriptions.Item label="节区总数">{sectionFeatures.section_num}</Descriptions.Item>
          </Descriptions>

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
        </Panel>
        
        <Panel header="字符串匹配详情" key="2">
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Card size="small" title="加密货币钱包">
                <p>比特币钱包数: {stringMatch.btc_count}</p>
                <p>莱特币钱包数: {stringMatch.ltc_count}</p>
                <p>门罗币钱包数: {stringMatch.xmr_count}</p>
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" title="网络地址">
                <p>URL数量: {stringMatch.urls_count}</p>
                <p>IP地址数: {stringMatch.ips_count}</p>
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" title="系统信息">
                <p>路径数量: {stringMatch.paths_count}</p>
                <p>注册表数量: {stringMatch.regs_count}</p>
              </Card>
            </Col>
          </Row>

          <Descriptions title="字符串匹配特征" column={3} style={{ marginTop: 16 }}>
            <Descriptions.Item label="MZ标记数量">{stringMatch.mz_count}</Descriptions.Item>
            <Descriptions.Item label="MZ平均长度">{stringMatch.mz_mean}</Descriptions.Item>
            <Descriptions.Item label="PE标记数量">{stringMatch.pe_count}</Descriptions.Item>
            <Descriptions.Item label="PE平均长度">{stringMatch.pe_mean}</Descriptions.Item>
            <Descriptions.Item label="矿池关键词数量">{stringMatch.pool_count}</Descriptions.Item>
            <Descriptions.Item label="矿池平均长度">{stringMatch.pool_mean}</Descriptions.Item>
            <Descriptions.Item label="CPU关键词数量">{stringMatch.cpu_count}</Descriptions.Item>
            <Descriptions.Item label="CPU平均长度">{stringMatch.cpu_mean}</Descriptions.Item>
            <Descriptions.Item label="GPU关键词数量">{stringMatch.gpu_count}</Descriptions.Item>
            <Descriptions.Item label="GPU平均长度">{stringMatch.gpu_mean}</Descriptions.Item>
            <Descriptions.Item label="数字货币关键词数量">{stringMatch.coin_count}</Descriptions.Item>
            <Descriptions.Item label="数字货币平均长度">{stringMatch.coin_mean}</Descriptions.Item>
          </Descriptions>

        </Panel>
        
        <Panel header="YARA规则匹配详情" key="3">
          <div>
            <Title level={5}>
              壳检测 {yaraMatch.packer_count > 0 ? 
                <Tag color="red"><WarningOutlined /> 检测到加壳</Tag> : 
                <Tag color="green"><CheckCircleOutlined /> 未检测到加壳</Tag>}
            </Title>
            <p>匹配加壳规则数量: {yaraMatch.packer_count}</p>
            
            <Title level={5}>
              恶意代码检测 {yaraMatch.yargen_count > 0 ? 
                <Tag color="red"><WarningOutlined /> 匹配恶意规则</Tag> : 
                <Tag color="green"><CheckCircleOutlined /> 未匹配恶意规则</Tag>}
            </Title>
            <p>匹配恶意规则数量: {yaraMatch.yargen_count}</p>
          </div>
        </Panel>
        
        <Panel header="字符串扫描详情" key="4">
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Statistic 
                title="杀毒软件检测" 
                value={stringCount.av_count} 
                prefix={<ExclamationCircleOutlined />} 
                valueStyle={{ color: stringCount.av_count > 0 ? '#cf1322' : '#3f8600' }}
              />
            </Col>
            <Col span={12}>
              <Statistic 
                title="调试器检测" 
                value={stringCount.dbg_count} 
                prefix={<ExclamationCircleOutlined />} 
                valueStyle={{ color: stringCount.dbg_count > 0 ? '#cf1322' : '#3f8600' }}
              />
            </Col>
            <Col span={12}>
              <Statistic 
                title="矿池名称" 
                value={stringCount.pool_name_count} 
                valueStyle={{ color: stringCount.pool_name_count > 5 ? '#cf1322' : '#3f8600' }}
              />
            </Col>
            <Col span={12}>
              <Statistic 
                title="加密算法" 
                value={stringCount.algorithm_name_count} 
                valueStyle={{ color: stringCount.algorithm_name_count > 2 ? '#cf1322' : '#3f8600' }}
              />
            </Col>
          </Row>
          <Card title="字符串匹配特征分析" style={{ marginTop: 16 }}>
            <ReactECharts option={stringMatchChartOption} style={{ height: 400 }} />
          </Card>
        </Panel>
        
        <Panel header="操作码分析详情" key="5">
          <Descriptions bordered size="small" column={3}>
            <Descriptions.Item label="最小操作码长度">{opcodeFeatures.opcode_min}</Descriptions.Item>
            <Descriptions.Item label="最大操作码长度">{opcodeFeatures.opcode_max}</Descriptions.Item>
            <Descriptions.Item label="操作码总和">{opcodeFeatures.opcode_sum}</Descriptions.Item>
            <Descriptions.Item label="操作码平均值">{opcodeFeatures.opcode_mean.toFixed(2)}</Descriptions.Item>
            <Descriptions.Item label="操作码方差">{opcodeFeatures.opcode_var.toFixed(2)}</Descriptions.Item>
            <Descriptions.Item label="操作码函数数量">{opcodeFeatures.opcode_count}</Descriptions.Item>
            <Descriptions.Item label="唯一操作码数量">{opcodeFeatures.opcode_uniq}</Descriptions.Item>
          </Descriptions>
        </Panel>
      </Collapse>
    </Card>
  );
};

export default FeatureEngineeringCard; 