import { GithubOutlined } from '@ant-design/icons';
import { DefaultFooter } from '@ant-design/pro-components';
import React from 'react';

const Footer: React.FC = () => {
  return (
    <DefaultFooter
      style={{
        background: 'none',
      }}
      copyright="Powered by zsh003"
      links={[
        {
          key: 'Author',
          title: 'Github',
          href: 'https://github.com/zsh003',
          blankTarget: true,
        },
        {
          key: 'github',
          title: <GithubOutlined />,
          href: 'https://github.com/zsh003/maltack',
          blankTarget: true,
        },
        {
          key: 'Maltact',
          title: 'Maltact',
          href: 'https://github.com/zsh003/maltack',
          blankTarget: true,
        }
      ]}
    />
  );
};

export default Footer;
