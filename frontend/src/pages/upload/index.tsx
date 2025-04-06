import React, { useState } from 'react';
import { Upload, Button, Typography, Card, Steps, Result, Spin, Row, Col, notification } from 'antd';
import { UploadOutlined, InboxOutlined, CheckCircleOutlined, LoadingOutlined, FileSearchOutlined } from '@ant-design/icons';
import { uploadSample } from '@/services/api';
import { history } from 'umi';

const { Title, Paragraph } = Typography;
const { Dragger } = Upload;
const { Step } = Steps;

const UploadPage: React.FC = () => {
  const [fileList, setFileList] = useState<any[]>([]);
  const [uploading, setUploading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async () => {
    const formData = new FormData();
    const file = fileList[0].originFileObj;
    formData.append('file', file);
    
    setUploading(true);
    setCurrentStep(1);
    
    try {
      const result = await uploadSample(file);
      setUploadResult(result);
      setCurrentStep(2);
      notification.success({
        message: '上传成功',
        description: '样本已成功上传并完成分析',
      });
    } catch (error) {
      console.error('上传失败', error);
      setError('上传或分析样本时遇到错误，请重试');
      setCurrentStep(3);
      notification.error({
        message: '上传失败',
        description: '上传或分析样本时遇到错误，请重试',
      });
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (info: any) => {
    setFileList(info.fileList.slice(-1));
    setCurrentStep(0);
    setUploadResult(null);
    setError(null);
  };

  const renderCurrentStepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <Card title="选择文件" className="upload-card">
            <Dragger
              name="file"
              fileList={fileList}
              onChange={handleFileChange}
              beforeUpload={() => false}
              accept=".exe,.dll,.sys,.ocx,.com"
              style={{ padding: '20px' }}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽PE文件到此区域上传</p>
              <p className="ant-upload-hint">
                支持的文件类型: exe, dll, sys, ocx, com
              </p>
            </Dragger>
            
            <div style={{ marginTop: '20px', textAlign: 'center' }}>
              <Button
                type="primary"
                onClick={handleUpload}
                disabled={fileList.length === 0}
                loading={uploading}
                size="large"
              >
                {uploading ? '上传中' : '开始上传'}
              </Button>
            </div>
          </Card>
        );
      case 1:
        return (
          <Card title="正在处理" className="upload-card">
            <div style={{ textAlign: 'center', padding: '40px' }}>
              <Spin indicator={<LoadingOutlined style={{ fontSize: 36 }} spin />} />
              <div style={{ marginTop: '20px' }}>
                <Title level={4}>正在分析样本...</Title>
                <Paragraph>正在上传并提取特征，这可能需要几分钟时间</Paragraph>
              </div>
            </div>
          </Card>
        );
      case 2:
        return (
          <Card title="分析完成" className="upload-card">
            <Result
              status="success"
              title="样本分析完成"
              subTitle={`样本ID: ${uploadResult.sample_id}`}
              extra={[
                <Button 
                  type="primary" 
                  key="detail" 
                  onClick={() => history.push(`/samples/${uploadResult.sample_id}`)}
                >
                  查看详情
                </Button>,
                <Button 
                  key="again" 
                  onClick={() => {
                    setFileList([]);
                    setCurrentStep(0);
                    setUploadResult(null);
                  }}
                >
                  上传新样本
                </Button>,
              ]}
            />
          </Card>
        );
      case 3:
        return (
          <Card title="上传失败" className="upload-card">
            <Result
              status="error"
              title="上传或分析失败"
              subTitle={error}
              extra={[
                <Button 
                  type="primary" 
                  key="again" 
                  onClick={() => {
                    setFileList([]);
                    setCurrentStep(0);
                    setError(null);
                  }}
                >
                  重新上传
                </Button>,
                <Button 
                  key="back" 
                  onClick={() => history.push('/samples')}
                >
                  返回样本列表
                </Button>,
              ]}
            />
          </Card>
        );
      default:
        return null;
    }
  };

  return (
    <div className="upload-page" style={{ padding: '24px' }}>
      <Title level={2}>上传样本</Title>
      <Paragraph>
        上传PE文件进行特征提取和恶意软件检测分析。上传后系统会自动提取直方图特征、PE静态特征和特征工程数据，并进行综合分析。
      </Paragraph>

      <Steps current={currentStep} style={{ marginBottom: '24px' }}>
        <Step title="选择文件" icon={<UploadOutlined />} />
        <Step title="特征提取" icon={<FileSearchOutlined />} />
        <Step title="分析结果" icon={<CheckCircleOutlined />} />
      </Steps>

      <Row justify="center">
        <Col xs={24} sm={20} md={16} lg={14} xl={12}>
          {renderCurrentStepContent()}
        </Col>
      </Row>

      <Card title="注意事项" style={{ marginTop: '24px' }}>
        <ul>
          <li>上传文件大小不超过50MB</li>
          <li>仅支持PE格式文件（exe, dll, sys等）</li>
          <li>不建议上传敏感文件（如包含个人隐私信息的文件）</li>
          <li>分析过程可能需要几分钟，请耐心等待</li>
          <li>本平台仅用于安全研究和教育目的</li>
        </ul>
      </Card>
    </div>
  );
};

export default UploadPage; 