import React from 'react';
import { Table, Button } from 'antd';
import { history } from '@umijs/max';
import useAnalysisModel from '@/models/analysis';

const UploadHistory = ({ history: uploadHistory }: { history: any[] }) => {
  const { updateCurrentFileId } = useAnalysisModel();

  const handleViewResult = (fileId: number) => {
    fileId = 31; // 测试用
    updateCurrentFileId(fileId);
    history.push(`/analysis/result/overview/${fileId}`);
  };

  return (
    <Table
      dataSource={uploadHistory}
      columns={[
        { title: '文件名称', dataIndex: 'file_name', key: 'file_name' },
        { title: '文件类型', dataIndex: 'file_type', key: 'file_type' },
        { title: '分析环境', dataIndex: 'environment', key: 'environment' },
        { title: '上传时间', dataIndex: 'upload_time', key: 'upload_time' },
        { title: '威胁等级', dataIndex: 'threat_level', key: 'threat_level' },
        { title: '状态', dataIndex: 'status', key: 'status' },
        {
          title: '操作',
          render: (_, record) => (
            <Button onClick={() => handleViewResult(record.id)}>
              查看分析结果
            </Button>
          )
        }
      ]}
      rowKey="id"
    />
  );
};

export default UploadHistory;
