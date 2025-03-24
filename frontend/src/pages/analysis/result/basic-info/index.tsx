import { useRequest } from '@umijs/max';
import { Card, Descriptions, Spin } from 'antd';
import type { FC } from 'react';
import { getBasicInfo } from '../service';

const BasicInfo: FC = () => {
  const { data, loading } = useRequest(() => getBasicInfo());

  if (loading) {
    return <Spin />;
  }

  const basicInfo = data?.basic_info || {};

  return (
    <Card title="文件基本信息">
      <Descriptions column={2}>
        <Descriptions.Item label="文件名">{basicInfo.file_name}</Descriptions.Item>
        <Descriptions.Item label="文件大小">{basicInfo.file_size} 字节</Descriptions.Item>
        <Descriptions.Item label="文件类型">{basicInfo.file_type}</Descriptions.Item>
        <Descriptions.Item label="MIME类型">{basicInfo.mime_type}</Descriptions.Item>
        <Descriptions.Item label="分析时间">{basicInfo.analyze_time}</Descriptions.Item>
        <Descriptions.Item label="MD5">{basicInfo.md5}</Descriptions.Item>
        <Descriptions.Item label="SHA1">{basicInfo.sha1}</Descriptions.Item>
        <Descriptions.Item label="SHA256">{basicInfo.sha256}</Descriptions.Item>
      </Descriptions>
    </Card>
  );
};

export default BasicInfo;
