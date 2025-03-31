import { Card, Spin, Empty, Divider, Button } from 'antd';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";
import { Bar } from '@ant-design/charts';

const HistogramFeatures: FC = () => {
  const { currentFileId } = useAnalysisModel();
  const [loading, setLoading] = useState(true);
  const [byteHistogram, setByteHistogram] = useState<any>(null);
  const [byteEntropy, setByteEntropy] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [histogramRes, entropyRes] = await Promise.all([
          axios.get(`http://localhost:5000/api/v1/analysis/features/histogram/${currentFileId}`),
          axios.get(`http://localhost:5000/api/v1/analysis/features/entropy/${currentFileId}`)
        ]);
        setByteHistogram(histogramRes.data);
        setByteEntropy(entropyRes.data);
      } catch (error) {
        console.error('获取直方图特征失败', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [currentFileId]);

  if (loading) {
    return <Spin />;
  }

  // 通用配置调整
  const COMMON_CONFIG = {
    //renderer: 'canvas' as const, // 启用Canvas渲染
    theme: 'dark',
    xAxis: {
      title: { text: '字节值 (0-255)', style: { fontSize: 14 } },
      min: 0,
      max: 255,
      tickCount: 16, // 显示更多刻度
      // label: {
      //   formatter: (val: number) => val % 16 === 0 ? val : '', // 每16个显示一个标签
      // }
    },
    interactions: [{ type: 'element-active' }, { type: 'brush' }], // 启用交互
    scrollbar: {
      x: {
        ratio: 0.2 // 横向滚动条
      }
    },
  };

  // 直方图配置修改
  const histogramConfig = {
    ...COMMON_CONFIG,
    data: (byteHistogram || [])
    .slice()
    .sort((a: { byte_value: number }, b: { byte_value: number }) => a.byte_value - b.byte_value),
    xField: 'byte_value',
    yField: 'count',
    yAxis: {
      title: { 
        text: '出现次数', 
        style: { fontSize: 14 },
        autoRotate: true 
      },
    },
    label: {
      position: 'right',
      style: { fill: '#666' }, // 更清晰的标签颜色
    },
  };

  // 熵值图配置修改
  const entropyConfig = {
    ...COMMON_CONFIG,
    data: (byteEntropy || [])
    .map((d: { [key: string]: any }) => ({
      ...d,
      entropy_value: Number(d.entropy_value.toFixed(2)) // 限制小数位数
    })),
    xField: 'byte_value',
    yField: 'entropy_value',
    lineStyle: {
      lineWidth: 1.5,
    },
    point: {
      size: 2,
      shape: 'circle',
    },
    tooltip: {
      fields: ['byte_value', 'entropy_value'],
      formatter: (item: any) => ({
        name: '熵值',
        value: item.entropy_value,
      }),
    },
  };

  // 添加导出功能
  const exportData = (type: 'csv' | 'json') => {
    const data = [...byteHistogram, ...byteEntropy];
    if (type === 'csv') {
      const csvContent = 'data:text/csv;charset=utf-8,'
        + 'byte_value,count,entropy_value\n'
        + data.map(d => `${d.byte_value},${d.count || ''},${d.entropy_value || ''}`).join('\n');
      window.open(encodeURI(csvContent));
    } else {
      const jsonStr = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      window.open(URL.createObjectURL(blob));
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <Card
        title="字节值分布直方图"
        extra={[
          <Button key="export-csv" onClick={() => exportData('csv')}>导出CSV</Button>,
          <Button key="export-json" onClick={() => exportData('json')}>导出JSON</Button>
        ]}
      >
        <div style={{ height: '80vh', position: 'relative' }}>
          <Bar {...histogramConfig} />
        </div>
      </Card>
      <Card
        title="字节熵分布直方图"
        extra={[
          <Button key="export-csv" onClick={() => exportData('csv')}>导出CSV</Button>,
          <Button key="export-json" onClick={() => exportData('json')}>导出JSON</Button>
        ]}
      >
        <div style={{ height: '80vh', position: 'relative' }}>
          <Bar {...entropyConfig} />
        </div>
      </Card>
    </div>
  );
};

export default HistogramFeatures;
