import React, { useState } from 'react';
import { Upload, Card, Col } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { beforeUpload, handleUploadSuccess, handleUploadError, handleUploadChange } from '../utils/index';

const FileUploader = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  return (
    <div className="file-uploader-wrapper">
      <Col span={20} offset={2}>
        <Card className="upload-card" title="文件上传">
          {/* 上传组件 */}
          <Upload.Dragger
            name="file"
            //action="http://localhost:5000/api/v1/analyze"
            // mock模拟
            customRequest={(options) => {
              import('../mock/uploadService').then(({ mockUpload }) => {
              mockUpload(options.file).then((response) => {
                options.onSuccess(response, options.file);
              }).catch((error) => {
                options.onError(error, options.file);
                });
              });
            }}


            beforeUpload={(file) => beforeUpload(file, setIsAnalyzing, setUploadError, setUploadLoading)}
            onSuccess={(response:unknown) => handleUploadSuccess(response, setIsAnalyzing, setUploadLoading, setUploadError)}
            onError={(file:unknown) => handleUploadError({file}, setIsAnalyzing, setUploadLoading, setUploadError)}
            onChange={handleUploadChange}
            loading={uploadLoading}
            showUploadList={false}
          >
            <p className="ant-upload-drag-icon">
              <UploadOutlined />
            </p>
            <p className="ant-upload-text">
              点击或拖拽文件到此处上传
            </p>
            {/* 状态提示 */}
            {isAnalyzing && (
              <div className="analysis-status">
                正在分析中，请稍候...
              </div>
            )}
            {uploadError && (
              <div className="upload-error">
                {uploadError}
              </div>
            )}
          </Upload.Dragger>
        </Card>
      </Col>
    </div>
  );
};

export default FileUploader;
