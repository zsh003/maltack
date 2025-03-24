import React from 'react';
import { InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { message, Upload } from 'antd';
import { history } from '@umijs/max';
import useAnalysisModel from '@/models/analysis';

const { Dragger } = Upload;

const App: React.FC = () => {
  const { updateCurrentFileId } = useAnalysisModel();

  const props: UploadProps = {
    name: 'file',
    multiple: true,
    action: 'http://localhost:5000/api/v1/upload',
    onChange(info) {
      const { status, response } = info.file;
      if (status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (status === 'done') {
        message.success(`${info.file.name} 文件上传成功`);
        // 获取上传后的文件ID并更新全局状态
        if (response && response.file_id) {
          updateCurrentFileId(response.file_id);
          history.push(`/analysis/result/overview/${response.file_id}`);
        }
      } else if (status === 'error') {
        message.error(`${info.file.name} 文件上传失败`);
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
    // defaultFileList: [
    //   {
    //     uid: '1',
    //     name: 'xxx.dll',
    //     size: 1234567, // 1.18 MB
    //     status: 'error',
    //     response: 'Server Error 500', // custom error message to show
    //     url: 'http://localhost:5000/upload/xxx.dll'
    //   },
    //   {
    //     uid: '2',
    //     name: 'yyy.exe',
    //     size: 123456789,  // 123456789 Bytes =  117.74 MB
    //     status: 'done',
    //     url: 'http://localhost:5000/upload/yyy.exe'
    //   },
    //   {
    //     uid: '3',
    //     name: 'zzz.elf',
    //     size: 1234567,
    //     status: 'done',
    //     url: 'http://localhost:5000/upload/zzz.elf'
    //   },
    // ],
    // 展示文件大小和删除按钮
    showUploadList: {
      extra: ({ size = 0 }) => (
        <span style={{ color: '#7c7c7c' }}>({(size / 1024 / 1024).toFixed(2)}MB)</span>
      ),
      showDownloadIcon: true,
      downloadIcon: "Download",
      showRemoveIcon: true,
    },
  };

  return (
    <Dragger {...props}>
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
      <p className="ant-upload-hint">
        支持单个或批量上传。严禁上传隐私数据或其他被禁止的文件。
      </p>
    </Dragger>
  );
};

export default App;
