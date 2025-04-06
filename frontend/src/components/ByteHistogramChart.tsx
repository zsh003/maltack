import React from 'react';
import ReactECharts from 'echarts-for-react';
import { Card } from 'antd';

interface ByteHistogramChartProps {
  byteHistogram: number[];
}

const ByteHistogramChart: React.FC<ByteHistogramChartProps> = ({ byteHistogram }) => {
  const options = {
    title: {
      text: '字节分布直方图',
      left: 'center',
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      formatter: function (params: any) {
        const value = params[0].value;
        const index = params[0].dataIndex;
        return `字节值: 0x${index.toString(16).padStart(2, '0')} (${index})<br/>频率: ${value}`;
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: Array.from({ length: 256 }, (_, i) => i),
      name: '字节值',
      axisLabel: {
        formatter: function (value: number) {
          if (value % 16 === 0) {
            return '0x' + value.toString(16).padStart(2, '0');
          }
          return '';
        },
      },
    },
    yAxis: {
      type: 'value',
      name: '频率',
    },
    series: [
      {
        name: '字节分布',
        type: 'bar',
        data: byteHistogram,
        itemStyle: {
          color: function(params: any) {
            // 为不同区域设置不同颜色
            if (params.dataIndex <= 31) { // 控制字符
              return '#ff7875';
            } else if (params.dataIndex >= 32 && params.dataIndex <= 127) { // ASCII可打印字符
              return '#52c41a';
            } else { // 扩展ASCII和高位字节
              return '#1890ff';
            }
          }
        }
      },
    ],
    dataZoom: [
      {
        type: 'slider',
        show: true,
        start: 0,
        end: 100,
      },
    ],
  };

  return (
    <Card title="字节直方图" className="chart-card">
      <ReactECharts option={options} style={{ height: '400px' }} />
    </Card>
  );
};

export default ByteHistogramChart; 