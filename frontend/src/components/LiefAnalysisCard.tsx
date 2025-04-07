import React from 'react';
import { Card, Table, Descriptions, Space, Tag, Collapse, Typography, Row, Col } from 'antd';
import type { LiefAnalysis } from '@/types/lief';

const { Panel } = Collapse;
const { Text } = Typography;

interface LiefAnalysisCardProps {
  liefData: LiefAnalysis;
}

const LiefAnalysisCard: React.FC<LiefAnalysisCardProps> = ({ liefData }) => {
  if (!liefData || Object.keys(liefData).length === 0) return null;

  const sectionsColumns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '虚拟地址',
      dataIndex: 'virtual_address',
      key: 'virtual_address',
      render: (addr: number) => `0x${addr.toString(16).toUpperCase()}`,
    },
    {
      title: '虚拟大小',
      dataIndex: 'virtual_size',
      key: 'virtual_size',
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`,
    },
    {
      title: '实际大小',
      dataIndex: 'size_of_raw_data',
      key: 'size_of_raw_data',
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`,
    },
    {
      title: '熵值',
      dataIndex: 'entropy',
      key: 'entropy',
      render: (entropy: number) => entropy.toFixed(2),
    },
    {
      title: '属性',
      key: 'permissions',
      render: (_: any, record: any) => (
        <Space>
          {record.is_readable && <Tag color="blue">可读</Tag>}
          {record.is_writable && <Tag color="orange">可写</Tag>}
          {record.is_executable && <Tag color="red">可执行</Tag>}
        </Space>
      ),
    },
  ];

  const importsColumns = [
    {
      title: '库名',
      dataIndex: 'library',
      key: 'library',
    },
    {
      title: '导入函数数量',
      dataIndex: 'entries',
      key: 'entries',
      render: (entries: any[]) => entries.length,
    },
  ];

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Row gutter={[16, 16]}>
        {liefData?.dos_header && (
          <Col span={12}>
            <Card title="DOS头信息">
              <Descriptions bordered column={1} size="small">
                <Descriptions.Item label="Magic">{liefData.dos_header.magic || 'N/A'}</Descriptions.Item>
                <Descriptions.Item label="文件大小(页)">{liefData.dos_header.file_size_in_pages || 'N/A'}</Descriptions.Item>
                <Descriptions.Item label="重定位表地址">0x{liefData.dos_header.address_of_relocation_table.toString(16).toUpperCase() || 'N/A'}</Descriptions.Item>
                <Descriptions.Item label="新PE头地址">0x{liefData.dos_header.address_of_new_exe_header.toString(16).toUpperCase() || 'N/A'}</Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
        )}
        {liefData?.header && (
          <Col span={12}>
            <Card title="PE头信息">
              <Descriptions bordered column={1} size="small">
                <Descriptions.Item label="签名">{liefData.header.signature || 'N/A'}</Descriptions.Item>
                <Descriptions.Item label="机器类型">{liefData.header.machine || 'N/A'}</Descriptions.Item>
                <Descriptions.Item label="节区数量">{liefData.header.number_of_sections || 'N/A'}</Descriptions.Item>
                <Descriptions.Item label="时间戳">{new Date(liefData.header.time_date_stamps * 1000).toLocaleString() || 'N/A'}</Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
        )}
      </Row>

      {liefData?.optioal_header && (
        <Card title="可选PE头信息">
          <Descriptions bordered column={2}>
            <Descriptions.Item label="Magic">0x{liefData.optional_header.magic || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="链接器版本">{`${liefData.optional_header.major_linker_version}.${liefData.optional_header.minor_linker_version}` || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="代码段大小">{(liefData.optional_header.size_of_code / 1024).toFixed(2) || 'N/A'} KB</Descriptions.Item>
            <Descriptions.Item label="初始化数据大小">{(liefData.optional_header.size_of_initialized_data / 1024).toFixed(2) || 'N/A'} KB</Descriptions.Item>
            <Descriptions.Item label="入口点">0x{liefData.optional_header.entry_point.toString(16).toUpperCase() || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="镜像基址">0x{liefData.optional_header.image_base.toString(16).toUpperCase() || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="子系统">{liefData.optional_header.subsystem || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="DLL特征值">
              <Space wrap>
                {liefData.optional_header.dll_characteristics.map((char: string) => (
                  <Tag key={char || 'N/A'} color="blue">{char || 'N/A'}</Tag>
                ))}
              </Space>
            </Descriptions.Item>
          </Descriptions>
        </Card>
      )}

      {liefData?.sections && (
        <Card title="节区信息">
          <Table
            dataSource={liefData.sections || []}
            columns={sectionsColumns}
            rowKey="name"
            pagination={false}
            size="small"
          />
        </Card>
      )}

      <Collapse>
        {liefData?.impoerts && (
          <Panel header="导入表" key="imports">
            <Table
              dataSource={liefData.imports}
              columns={importsColumns}
              rowKey="library"
              expandable={{
                expandedRowRender: (record) => (
                  <Table
                    dataSource={record.entries}
                    columns={[
                      { title: '函数名', dataIndex: 'name', key: 'name' },
                      { title: 'Hint', dataIndex: 'hint', key: 'hint' },
                    ]}
                    pagination={false}
                    rowKey="name"
                    size="small"
                  />
                ),
              }}
            />
          </Panel>
        )}

        {liefData?.exports && (
          <Panel header="导出表" key="exports">
            <Table
              dataSource={liefData.exports}
              columns={[
                { title: '函数名', dataIndex: 'name', key: 'name' },
                { title: '地址', dataIndex: 'address', key: 'address' },
                {
                  title: '转发',
                  dataIndex: 'is_forwarded',
                  key: 'is_forwarded',
                  render: (forwarded: boolean) => forwarded ? '是' : '否',
                },
              ]}
              rowKey="name"
              size="small"
            />
          </Panel>
        )}

        {liefData?.tls && (
          <Panel header="TLS信息" key="tls">
            <Descriptions bordered column={1}>
              <Descriptions.Item label="回调函数数量">{liefData.tls.callbacks}</Descriptions.Item>
              <Descriptions.Item label="原始数据地址范围">
                {`0x${liefData.tls.address_of_raw_data[0].toString(16).toUpperCase()} - 0x${liefData.tls.address_of_raw_data[1].toString(16).toUpperCase()}`}
              </Descriptions.Item>
              <Descriptions.Item label="索引地址">
                {`0x${liefData.tls.address_of_index.toString(16).toUpperCase()}`}
              </Descriptions.Item>
              <Descriptions.Item label="零填充大小">{liefData.tls.size_of_zero_fill} bytes</Descriptions.Item>
            </Descriptions>
          </Panel>
        )}

        {liefData?.resources && liefData.resources.length > 0 && (
          <Panel header="资源信息" key="resources">
            <Table
              dataSource={liefData.resources}
              columns={[
                { title: '类型', dataIndex: 'type', key: 'type' },
                { title: 'ID', dataIndex: 'id', key: 'id' },
                { title: '语言', dataIndex: 'language', key: 'language' },
                { title: '子语言', dataIndex: 'sublanguage', key: 'sublanguage' },
              ]}
              expandable={{
                expandedRowRender: (record) => record.content && (
                  <pre style={{ maxHeight: '200px', overflow: 'auto' }}>
                    {record.content}
                  </pre>
                ),
              }}
              rowKey={(record) => `${record.type}-${record.id}-${record.language}`}
              size="small"
            />
          </Panel>
        )}

        {liefData?.manifest && (
          <Panel header="清单文件" key="manifest">
            <pre style={{ maxHeight: '400px', overflow: 'auto' }}>
              {liefData.manifest}
            </pre>
          </Panel>
        )}

        {liefData.signatures?.has_signature && (
          <Panel header="签名信息" key="signatures">
            <Descriptions bordered column={1}>
              <Descriptions.Item label="版本">{liefData.signatures.signature_info?.version}</Descriptions.Item>
              <Descriptions.Item label="摘要算法">{liefData.signatures.signature_info?.digest_algorithm}</Descriptions.Item>
              <Descriptions.Item label="内容类型">{liefData.signatures.signature_info?.content_info.content_type}</Descriptions.Item>
              <Descriptions.Item label="内容摘要算法">{liefData.signatures.signature_info?.content_info.digest_algorithm}</Descriptions.Item>
            </Descriptions>
          </Panel>
        )}
      </Collapse>
    </Space>
  );
};

export default LiefAnalysisCard; 