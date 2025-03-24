import { PageContainer } from '@ant-design/pro-components';
import { history, Outlet, useLocation, useMatch, useParams } from '@umijs/max';
import { Input } from 'antd';
import type { FC } from 'react';
import { useEffect } from 'react';
import useAnalysisModel from '@/models/analysis';

type SearchProps = {
  children?: React.ReactNode;
};

const tabList = [
  {
    key: 'overview',
    tab: '分析结果概要',
  },
  {
    key: 'basic-info',
    tab: '文件基本信息',
  },
  {
    key: 'pe-info',
    tab: 'PE文件信息',
  },
  {
    key: 'yara-rules',
    tab: 'Yara规则匹配',
  },
  {
    key: 'sigma-rules',
    tab: 'Sigma规则匹配',
  },
  {
    key: 'strings',
    tab: '字符串分析',
  },
];

const Search: FC<SearchProps> = () => {
  const location = useLocation();
  const { fileId } = useParams<{ fileId: string }>();
  const { currentFileId, updateCurrentFileId } = useAnalysisModel();
  const match = useMatch(location.pathname);

  useEffect(() => {
    if (fileId) {
      updateCurrentFileId(parseInt(fileId, 10));
    }
  }, [fileId, updateCurrentFileId]);

  const handleTabChange = (key: string) => {
    const targetPath = `/analysis/result/${key}/${currentFileId}`;
    history.push(targetPath);
  };

  const handleFormSubmit = (value: string) => {
    // eslint-disable-next-line no-console
    console.log(value);
  };

  const getTabKey = () => {
    const pathParts = location.pathname.split('/');
    const resultIndex = pathParts.indexOf('result');
    const tabKey = resultIndex !== -1 ? pathParts[resultIndex + 1] : 'overview';

    // 校验是否为有效标签项
    return tabList.some(item => item.key === tabKey)
      ? tabKey
      : 'overview';
  };

  return (
    <PageContainer
      content={
        <div style={{ textAlign: 'center' }}>
          <Input.Search
            placeholder="请输入"
            enterButton="搜索"
            size="large"
            onSearch={handleFormSubmit}
            style={{ maxWidth: 522, width: '100%' }}
          />
        </div>
      }
      tabList={tabList}
      tabActiveKey={getTabKey()}
      onTabChange={handleTabChange}
    >
      <Outlet />
    </PageContainer>
  );
};

export default Search;
