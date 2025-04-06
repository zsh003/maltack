import React, { useEffect, useState } from 'react';
import { useParams } from 'umi';
import { Spin, Typography, Descriptions, Card, Tag, Tabs, Result, Row, Col, Divider } from 'antd';
import { FileTextOutlined, WarningOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { fetchSampleDetail } from '@/services/api';
import ByteHistogramChart from '@/components/ByteHistogramChart';
import EntropyHistogramChart from '@/components/EntropyHistogramChart';
import PEFeaturesCard from '@/components/PEFeaturesCard';
import FeatureEngineeringCard from '@/components/FeatureEngineeringCard';

const { Title, Paragraph } = Typography;
const { TabPane } = Tabs;

const SampleDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [loading, setLoading] = useState(true);
  const [sample, setSample] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadSample = async () => {
      setLoading(true);
      try {
        const data = await fetchSampleDetail(parseInt(id));
        setSample(data);
      } catch (error) {
        console.error('加载样本详情失败', error);
        setError('加载样本详情失败，请稍后重试');
      } finally {
        setLoading(false);
      }
    };

    loadSample();
  }, [id]);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px' }}>
        <Spin size="large" tip="正在加载样本详情..." />
      </div>
    );
  }

  if (error || !sample) {
    return (
      <Result
        status="error"
        title="加载失败"
        subTitle={error || '未能找到指定样本'}
      />
    );
  }

  return (
    <div className="sample-detail-page" style={{ padding: '24px' }}>
      <div className="page-header">
        <Title level={3}>{sample.file_name}</Title>
        <Tag color={sample.is_malicious === 1 ? 'error' : 'success'} style={{ fontSize: '14px', padding: '4px 8px' }}>
          {sample.is_malicious === 1 
            ? <><WarningOutlined /> 恶意软件</>
            : <><CheckCircleOutlined /> 正常软件</>}
        </Tag>
      </div>

      <Card style={{ marginTop: '16px' }}>
        <Descriptions title="基本信息" bordered>
          <Descriptions.Item label="样本ID">{sample.id}</Descriptions.Item>
          <Descriptions.Item label="文件名">{sample.file_name}</Descriptions.Item>
          <Descriptions.Item label="文件大小">{(sample.file_size / 1024).toFixed(2)} KB</Descriptions.Item>
          <Descriptions.Item label="MD5哈希" span={3}>{sample.file_hash}</Descriptions.Item>
          <Descriptions.Item label="分析时间">{sample.analysis_time}</Descriptions.Item>
          <Descriptions.Item label="分类结果" span={2}>
            <Tag color={sample.is_malicious === 1 ? 'error' : 'success'}>
              {sample.classification_result}
            </Tag>
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Tabs defaultActiveKey="1" style={{ marginTop: '24px' }}>
        <TabPane 
          tab={<span><FileTextOutlined />直方图特征</span>}
          key="1"
        >
          {sample.histogram_features ? (
            <div>
              <Paragraph>
                直方图特征是基于文件字节统计分析的特征，包括字节分布直方图和字节熵直方图。
                这类特征可以有效检测加密/填充数据，识别代码段和混淆区域。
              </Paragraph>
              
              <Divider orientation="left">特征图表</Divider>
              
              <Row gutter={[16, 16]}>
                <Col span={24}>
                  <ByteHistogramChart 
                    byteHistogram={sample.histogram_features.byte_histogram} 
                  />
                </Col>
                <Col span={24}>
                  <EntropyHistogramChart 
                    entropyHistogram={sample.histogram_features.entropy_histogram} 
                  />
                </Col>
              </Row>
            </div>
          ) : (
            <Result
              status="warning"
              title="暂无直方图特征数据"
              subTitle="未能提取该样本的直方图特征"
            />
          )}
        </TabPane>

        <TabPane 
          tab={<span><FileTextOutlined />PE静态特征</span>}
          key="2"
        >
          {sample.pe_features ? (
            <div>
              <Paragraph>
                PE静态特征是从PE文件结构中提取的特征，包括文件头信息、节区属性、导入导出表等。
                这类特征可以有效识别异常PE结构，检测壳和代码注入等恶意行为。
              </Paragraph>
              
              <PEFeaturesCard 
                generalInfo={sample.pe_features.general_info}
                headerInfo={sample.pe_features.header_info}
                sectionInfo={sample.pe_features.section_info}
                exportsInfo={sample.pe_features.exports_info}
              />
            </div>
          ) : (
            <Result
              status="warning"
              title="暂无PE静态特征数据"
              subTitle="未能提取该样本的PE静态特征"
            />
          )}
        </TabPane>

        <TabPane 
          tab={<span><FileTextOutlined />特征工程</span>}
          key="3"
        >
          {sample.engineered_features ? (
            <div>
              <Paragraph>
                特征工程是基于专家领域知识提取的高级特征，包括节区信息、字符串匹配、Yara规则匹配、关键字扫描和操作码分析等。
                这类特征可以有效检测挖矿软件、加壳程序和其他高级恶意行为。
              </Paragraph>
              
              <FeatureEngineeringCard 
                sectionFeatures={sample.engineered_features.section_features}
                stringMatch={sample.engineered_features.string_match}
                yaraMatch={sample.engineered_features.yara_match}
                stringCount={sample.engineered_features.string_count}
                opcodeFeatures={sample.engineered_features.opcode_features}
              />
            </div>
          ) : (
            <Result
              status="warning"
              title="暂无特征工程数据"
              subTitle="未能提取该样本的特征工程数据"
            />
          )}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default SampleDetailPage; 