import React from 'react';
import { Progress } from 'antd';

interface ProgressBarProps {
  progress: number;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ progress }) => {
  return <Progress percent={progress} status={progress === 100 ? 'success' : 'active'} />;
};

export default ProgressBar;
