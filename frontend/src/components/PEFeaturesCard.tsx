import React from 'react';
import { Card, Table, Descriptions, Typography, Tag, Badge } from 'antd';
import { WarningOutlined, CheckCircleOutlined, FileOutlined } from '@ant-design/icons';

const { Title } = Typography;

interface PESection {
  name: string;
  size: number;
  entropy: number;
  vsize: number;
  props: string[];
}

interface PEFeaturesCardProps {
  generalInfo: any;
  headerInfo: any;
  sectionInfo: {
    entry: string;
    sections: PESection[];
  };
  exportsInfo: any;
}

const PEFeaturesCard: React.FC<PEFeaturesCardProps> = ({ 
  generalInfo, 
  headerInfo, 
  sectionInfo, 
  exportsInfo 
}) => {
  // 处理节区信息的表格
  const sectionColumns = [
    {
      title: '节区名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string) => (
        <span>
          <FileOutlined style={{ marginRight: 8 }} />
          {text}
        </span>
      ),
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`,
    },
    {
      title: '虚拟大小',
      dataIndex: 'vsize',
      key: 'vsize',
      render: (vsize: number) => `${(vsize / 1024).toFixed(2)} KB`,
    },
    {
      title: '熵值',
      dataIndex: 'entropy',
      key: 'entropy',
      render: (entropy: number) => {
        let color = 'green';
        let icon = <CheckCircleOutlined />;
        
        if (entropy > 7.0) {
          color = 'red';
          icon = <WarningOutlined />;
        } else if (entropy > 6.0) {
          color = 'orange';
          icon = <WarningOutlined />;
        }
        
        return (
          <span>
            <Badge status={color as any} />
            {entropy.toFixed(2)} {icon}
          </span>
        );
      },
    },
    {
      title: '属性',
      dataIndex: 'props',
      key: 'props',
      render: (props: string[]) => (
        <>
          {props.map(prop => {
            let color = 'blue';
            if (prop.includes('EXECUTE')) {
              color = 'volcano';
            } else if (prop.includes('WRITE')) {
              color = 'orange';
            } else if (prop.includes('READ')) {
              color = 'green';
            }
            return (
              <Tag color={color} key={prop}>
                {prop}
              </Tag>
            );
          })}
        </>
      ),
    },
  ];

  return (
    <Card title="PE文件结构特征" className="pe-features-card">
      <Title level={4}>常规信息</Title>
      <Descriptions bordered column={2}>
        <Descriptions.Item label="主版本号">{generalInfo.major_version}</Descriptions.Item>
        <Descriptions.Item label="次版本号">{generalInfo.minor_version}</Descriptions.Item>
        <Descriptions.Item label="调试信息大小">{generalInfo.debug_size} 字节</Descriptions.Item>
        <Descriptions.Item label="TLS区段大小">{generalInfo.tls_size} 字节</Descriptions.Item>
        <Descriptions.Item label="重定位信息大小">{generalInfo.relocations_size} 字节</Descriptions.Item>
        <Descriptions.Item label="数据目录数量">{generalInfo.num_data_directories}</Descriptions.Item>
      </Descriptions>

      <Title level={4} style={{ marginTop: 16 }}>PE头信息</Title>
      <Descriptions bordered column={2}>
        <Descriptions.Item label="PE签名">{headerInfo.pe_signature}</Descriptions.Item>
        <Descriptions.Item label="机器类型">{headerInfo.machine_type}</Descriptions.Item>
        <Descriptions.Item label="时间戳">{headerInfo.timestamp}</Descriptions.Item>
        <Descriptions.Item label="节区数量">{headerInfo.num_sections}</Descriptions.Item>
        <Descriptions.Item label="特征值">
          {headerInfo.characteristics.map((char: string) => (
            <Tag key={char}>{char}</Tag>
          ))}
        </Descriptions.Item>
      </Descriptions>

      <Title level={4} style={{ marginTop: 16 }}>节区信息</Title>
      <p>入口点所在节区: <Tag color="purple">{sectionInfo.entry}</Tag></p>
      <Table 
        dataSource={sectionInfo.sections} 
        columns={sectionColumns} 
        rowKey="name"
        pagination={false}
        size="small"
      />

      {exportsInfo.exports && exportsInfo.exports.length > 0 && (
        <>
          <Title level={4} style={{ marginTop: 16 }}>导出函数</Title>
          <Table 
            dataSource={exportsInfo.exports} 
            columns={[
              { title: '函数名', dataIndex: 'name', key: 'name' },
              { title: '地址', dataIndex: 'address', key: 'address' },
            ]} 
            rowKey="name"
            pagination={false}
            size="small"
          />
        </>
      )}
    </Card>
  );
};

export default PEFeaturesCard; 