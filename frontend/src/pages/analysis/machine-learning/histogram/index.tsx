import { Card, Spin } from 'antd';
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

  const histogramConfig = {
    data: byteHistogram || [],
    xField: 'byte_value',
    yField: 'count',
    label: {
      position: 'top',
    },
    xAxis: {
      title: { text: '字节值 (0-255)' },
      min: 0,
      max: 255,
    },
    yAxis: {
      title: { text: '出现次数' },
    },
  };

  const entropyConfig = {
    data: byteEntropy || [],
    xField: 'byte_value',
    yField: 'entropy_value',
    label: {
      position: 'top',
    },
    xAxis: {
      title: { text: '字节值 (0-255)' },
      min: 0,
      max: 255,
    },
    yAxis: {
      title: { text: '熵值' },
    },
  };

  return (
    <div>
      <Card title="字节直方图特征" style={{ marginBottom: 16 }}>
        <Bar {...histogramConfig} />
      </Card>
      <Card title="字节熵直方图特征">
        <Bar {...entropyConfig} />
      </Card>
    </div>
  );
};

export default HistogramFeatures;
