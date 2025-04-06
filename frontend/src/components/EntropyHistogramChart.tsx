import React from 'react';
import ReactECharts from 'echarts-for-react';
import { Card } from 'antd';

interface EntropyHistogramChartProps {
  entropyHistogram: number[];
}

const EntropyHistogramChart: React.FC<EntropyHistogramChartProps> = ({ entropyHistogram }) => {
  const options = {
    title: {
      text: '熵值直方图',
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
        return `熵窗口: 0x${index.toString(16).padStart(2, '0')} (${index})<br/>数量: ${value}`;
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
      name: '熵窗口',
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
      name: '数量',
    },
    series: [
      {
        name: '熵分布',
        type: 'bar',
        data: entropyHistogram,
        itemStyle: {
          color: function(params: any) {
            const value = params.value;
            // 熵值越高，颜色越红
            if (value > 5000) {
              return '#f5222d'; // 高熵值区域（疑似加密/压缩）
            } else if (value > 1000) {
              return '#fa8c16'; // 中熵值
            } else {
              return '#52c41a'; // 低熵值区域（一般代码）
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
    <Card title="熵值直方图" className="chart-card">
      <ReactECharts option={options} style={{ height: '400px' }} />
    </Card>
  );
};

export default EntropyHistogramChart; 