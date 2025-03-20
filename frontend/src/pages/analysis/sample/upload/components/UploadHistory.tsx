import React from 'react';
import { Table, Button } from 'antd';

const UploadHistory = ({ history }: { history: any[] }) => {
  return (
    <Table
      dataSource={history}
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
            <Button onClick={() => { alert('查看分析结果'); }}>查看分析结果</Button>
          )
        }
      ]}
      rowKey="id"
    />
  );
};

export default UploadHistory;
