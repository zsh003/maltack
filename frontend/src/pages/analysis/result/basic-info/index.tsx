import { useRequest } from '@umijs/max';
import { useState, useEffect } from 'react';
import { Card, Descriptions, Spin } from 'antd';
import type { FC } from 'react';
import { getBasicInfo } from '../service';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";

const BasicInfo: FC = () => {
  const { currentFileId } = useAnalysisModel();

  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get(`http://localhost:5000/api/v1/analysis/result/basic-info/${currentFileId}`);
        setData(data.basic_info);
      } catch (error) {
        console.error('获取上传历史失败', error);
      }
    };

    fetchData();
  }, []);

  return (
    <Card title="文件基本信息">
      <Descriptions column={2}>
        <Descriptions.Item label="文件名">{data?.file_name}</Descriptions.Item>
        <Descriptions.Item label="文件大小">{data?.file_size} 字节</Descriptions.Item>
        <Descriptions.Item label="文件类型">{data?.file_type}</Descriptions.Item>
        <Descriptions.Item label="MIME类型">{data?.mime_type}</Descriptions.Item>
        <Descriptions.Item label="分析时间">{data?.analyze_time}</Descriptions.Item>
        <Descriptions.Item label="MD5">{data?.md5}</Descriptions.Item>
        <Descriptions.Item label="SHA1">{data?.sha1}</Descriptions.Item>
        <Descriptions.Item label="SHA256">{data?.sha256}</Descriptions.Item>
      </Descriptions>
    </Card>
  );
};

export default BasicInfo;
