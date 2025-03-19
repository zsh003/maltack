import { PageContainer } from '@ant-design/pro-components';
import { history, Outlet, useLocation, useMatch } from '@umijs/max';
import { Input } from 'antd';
import type { FC } from 'react';

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
  let match = useMatch(location.pathname);
  const handleTabChange = (key: string) => {
    const url =
      match?.pathname === '/' ? '' : match?.pathname.substring(0, match.pathname.lastIndexOf('/'));
    switch (key) {
      case 'overview':
        history.push(`${url}/overview`);
        break;
      case 'basic-info':
        history.push(`${url}/basic-info`);
        break;
      case 'yara-rules':
        history.push(`${url}/yara-rules`);
        break;
      case 'sigma-rules':
        history.push(`${url}/sigma-rules`);
        break;
      case 'strings':
        history.push(`${url}/strings`);
        break;
      default:
        break;
    }
  };

  const handleFormSubmit = (value: string) => {
    // eslint-disable-next-line no-console
    console.log(value);
  };

  const getTabKey = () => {
    const tabKey = location.pathname.substring(location.pathname.lastIndexOf('/') + 1);
    if (tabKey && tabKey !== '/') {
      return tabKey;
    }
    return 'overview';
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
