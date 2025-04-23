import React from 'react';
import { Card, Table, Descriptions, Space, Row, Col, Progress, Statistic } from 'antd';
import ReactECharts from 'echarts-for-react';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
}

interface ModelAnalysis {
  histogram_model: {
    metrics: ModelMetrics;
    training_history: {
      epoch: number;
      loss: number;
      accuracy: number;
      val_loss: number;
      val_accuracy: number;
    }[];
  };
  pe_raw_models: {
    model_name: string;
    accuracy: number;
    importance: {
      feature: string;
      score: number;
    }[];
  }[];
  feature_engineering: {
    metrics: ModelMetrics;
    feature_importance: {
      feature: string;
      importance: number;
    }[];
  };
  ensemble_metrics: ModelMetrics;
}

interface ModelAnalysisCardProps {
  data: ModelAnalysis;
}

const ModelAnalysisCard: React.FC<ModelAnalysisCardProps> = ({ data }) => {
  if (!data) return null;

  const renderConfusionMatrix = (matrix: number[][]) => {
    const option = {
      tooltip: {
        trigger: 'item'
      },
      legend: {
        top: '5%',
        left: 'center'
      },
      series: [
        {
          name: '混淆矩阵',
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: true,
            formatter: '{b}: {c} ({d}%)'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '16',
              fontWeight: 'bold'
            }
          },
          labelLine: {
            show: true
          },
          data: [
            { value: matrix[0][0], name: '真阴性' },
            { value: matrix[0][1], name: '假阳性' },
            { value: matrix[1][0], name: '假阴性' },
            { value: matrix[1][1], name: '真阳性' }
          ]
        }
      ]
    };

    return <ReactECharts option={option} style={{ height: '400px' }} />;
  };

  const renderTrainingHistory = (history: any[]) => {
    const option = {
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: ['训练损失', '训练准确率', '验证损失', '验证准确率']
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: history.map(h => h.epoch)
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          name: '训练损失',
          type: 'line',
          smooth: true,
          data: history.map(h => h.loss)
        },
        {
          name: '训练准确率',
          type: 'line',
          smooth: true,
          data: history.map(h => h.accuracy)
        },
        {
          name: '验证损失',
          type: 'line',
          smooth: true,
          data: history.map(h => h.val_loss)
        },
        {
          name: '验证准确率',
          type: 'line',
          smooth: true,
          data: history.map(h => h.val_accuracy)
        }
      ]
    };

    return <ReactECharts option={option} style={{ height: '300px' }} />;
  };

  const renderFeatureImportance = (importance: any[]) => {
    const sortedData = importance
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 20);

    const option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'value',
        name: '重要性'
      },
      yAxis: {
        type: 'category',
        data: sortedData.map(item => item.feature),
        inverse: true
      },
      series: [
        {
          name: '特征重要性',
          type: 'bar',
          data: sortedData.map(item => item.importance),
          label: {
            show: true,
            position: 'right',
            formatter: '{c}'
          }
        }
      ]
    };

    return <ReactECharts option={option} style={{ height: '600px' }} />;
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="集成学习整体性能">
            <Row gutter={16}>
              <Col span={6}>
                <Statistic 
                  title="整体准确率" 
                  value={data.ensemble_metrics.accuracy} 
                  suffix="%" 
                  precision={2}
                />
                <Progress 
                  percent={data.ensemble_metrics.accuracy} 
                  status="active"
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="精确率" 
                  value={data.ensemble_metrics.precision} 
                  suffix="%" 
                  precision={2}
                />
                <Progress 
                  percent={data.ensemble_metrics.precision} 
                  status="active"
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="召回率" 
                  value={data.ensemble_metrics.recall} 
                  suffix="%" 
                  precision={2}
                />
                <Progress 
                  percent={data.ensemble_metrics.recall} 
                  status="active"
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="F1分数" 
                  value={data.ensemble_metrics.f1_score} 
                  suffix="%" 
                  precision={2}
                />
                <Progress 
                  percent={data.ensemble_metrics.f1_score} 
                  status="active"
                />
              </Col>
            </Row>
            <div style={{ marginTop: 24 }}>
              <h4>混淆矩阵</h4>
              {renderConfusionMatrix(data.ensemble_metrics.confusion_matrix)}
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="直方图CNN模型训练过程">
            {data.histogram_model.training_history && renderTrainingHistory(data.histogram_model.training_history)}
            <Descriptions bordered column={2} size="small" style={{ marginTop: 16 }}>
              <Descriptions.Item label="准确率">
                {data.histogram_model.metrics.accuracy.toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="精确率">
                {data.histogram_model.metrics.precision.toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="召回率">
                {data.histogram_model.metrics.recall.toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="F1分数">
                {data.histogram_model.metrics.f1_score.toFixed(2)}%
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="PE静态特征模型性能">
            <Table
              dataSource={data.pe_raw_models}
              columns={[
                { title: '模型', dataIndex: 'model_name', key: 'model_name' },
                { 
                  title: '准确率', 
                  dataIndex: 'accuracy', 
                  key: 'accuracy',
                  render: (acc: number) => `${acc.toFixed(2)}%`
                },
              ]}
              size="small"
              pagination={false}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="特征工程模型分析">
            <Row gutter={16}>
              <Col span={12}>
                <h4>模型性能指标</h4>
                <Descriptions bordered column={2} size="small">
                  <Descriptions.Item label="准确率">
                    {data.feature_engineering.metrics.accuracy.toFixed(2)}%
                  </Descriptions.Item>
                  <Descriptions.Item label="精确率">
                    {data.feature_engineering.metrics.precision.toFixed(2)}%
                  </Descriptions.Item>
                  <Descriptions.Item label="召回率">
                    {data.feature_engineering.metrics.recall.toFixed(2)}%
                  </Descriptions.Item>
                  <Descriptions.Item label="F1分数">
                    {data.feature_engineering.metrics.f1_score.toFixed(2)}%
                  </Descriptions.Item>
                </Descriptions>
              </Col>
              <Col span={12}>
                <h4>特征重要性排名</h4>
                {/* {data.feature_engineering.feature_importance && 
                  renderFeatureImportance(data.feature_engineering.feature_importance)} */}
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </Space>
  );
};

export default ModelAnalysisCard; 