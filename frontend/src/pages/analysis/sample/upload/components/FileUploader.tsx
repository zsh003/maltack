import React from 'react';
import { InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { message, Upload } from 'antd';

const { Dragger } = Upload;

const props: UploadProps = {
  name: 'file',
  multiple: true,
  action: 'https://localhost:5000/api/v1/upload',
  onChange(info) {

    const { status } = info.file;
    if (status !== 'uploading') {
      console.log(info.file, info.fileList);
    }
    if (status === 'done') {
      message.success(`${info.file.name} file uploaded successfully.`);
    } else if (status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
  // 自定义进度条
  progress: {
    strokeColor: {
      '0%': '#108ee9',
      '100%': '#87d068',
    },
    strokeWidth: 3,
    format: (percent) => percent && `${parseFloat(percent.toFixed(2))}%`,
},
  onDrop(e) {
    console.log('Dropped files', e.dataTransfer.files);
  },
  defaultFileList: [
    {
      uid: '1',
      name: 'xxx.dll',
      size: 1234567, // 1.18 MB
      status: 'error',
      response: 'Server Error 500', // custom error message to show
      url: 'http://localhost:5000/upload/xxx.dll'
    },
    {
      uid: '2',
      name: 'yyy.exe',
      size: 123456789,  // 123456789 Bytes =  117.74 MB
      status: 'done',
      url: 'http://localhost:5000/upload/yyy.exe'
    },
    {
      uid: '3',
      name: 'zzz.elf',
      size: 1234567,
      status: 'done',
      url: 'http://localhost:5000/upload/zzz.elf'
    },
  ],
  // 展示文件大小和删除按钮
  showUploadList: {
    extra: ({ size = 0 }) => (
      <span style={{ color: '#7c7c7c' }}>({(size / 1024 / 1024).toFixed(2)}MB)</span>
    ),
    showRemoveIcon: true,
  },
};

const App: React.FC = () => (
  <Dragger {...props}>
    <p className="ant-upload-drag-icon">
      <InboxOutlined />
    </p>
    <p className="ant-upload-text">Click or drag file to this area to upload</p>
    <p className="ant-upload-hint">
      Support for a single or bulk upload. Strictly prohibited from uploading company data or other
      banned files.
    </p>
  </Dragger>
);

export default App;
