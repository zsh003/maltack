import React, { useState } from 'react';
import { Upload, Progress } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import {beforeUpload, handleUploadError, handleUploadSuccess} from '../utils/uploadUtils';  // 引入工具函数
import ProgressBar from './ProgressBar';  // 引入进度条组件
import { handleFileUpload } from '../services/apiService';  // 上传接口请求

const FileUploader = () => {
  const [uploadProgress, setUploadProgress] = useState(0);  // 上传进度
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  // 修改beforeUpload调用
  const beforeUploadWrapper = (file: File) => {
    return beforeUpload(
      file,
      setIsAnalyzing,
      setUploadError,
      setUploadLoading
    );
  };

  const handleUpload = (options: any) => {
    const handleProgress = (percent: number) => {
      setUploadProgress(percent);
    };

    handleFileUpload(options.file, handleProgress)
      .then((response) => {
        setUploadProgress(100);
        handleUploadSuccess(response, setIsAnalyzing, setUploadLoading, setUploadError);
      })
      .catch((error) => {
        handleUploadError({ file: options.file }, setIsAnalyzing, setUploadLoading, setUploadError);
      });
  };

  return (
      <div className="file-uploader">
        <Upload.Dragger
            name="file"
            customRequest={handleUpload}
            beforeUpload={beforeUploadWrapper}
            onProgress={({ percent }) => setUploadProgress(percent)}  // 更新上传进度
            className="upload-dragger"
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件上传</p>
        </Upload.Dragger>

        {/* 进度条显示 */}
        <ProgressBar progress={uploadProgress} />
        {uploadError && <div className="error-message">{uploadError}</div>}
      </div>
  );
};

export default FileUploader;
