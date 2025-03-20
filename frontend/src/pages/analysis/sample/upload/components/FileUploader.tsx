import React, { useState } from 'react';
import { Upload, Progress } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { beforeUpload } from '../utils/uploadUtils';  // 引入工具函数
import ProgressBar from './ProgressBar';  // 引入进度条组件
import { handleFileUpload } from '../services/apiService';  // 上传接口请求

const FileUploader = () => {
  const [uploadProgress, setUploadProgress] = useState(0); // 上传进度

  const handleUpload = (options: any) => {
    handleFileUpload(options.file)
        .then(response => {
          setUploadProgress(100); // 上传完成
        })
        .catch(error => {
          // 错误处理
        });
  };

  return (
      <div className="file-uploader">
        <Upload.Dragger
            name="file"
            customRequest={handleUpload}
            beforeUpload={(file) => beforeUpload(file)}
            onProgress={({ percent }) => setUploadProgress(percent)}  // 更新上传进度
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件上传</p>
        </Upload.Dragger>

        {/* 进度条显示 */}
        <ProgressBar progress={uploadProgress} />
      </div>
  );
};

export default FileUploader;
