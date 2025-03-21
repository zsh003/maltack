import React from 'react';
import { Progress } from 'antd';

interface ProgressBarProps {
  progress: number;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ progress }) => {
  const status = progress === 100 ? 'success' : progress > 0 ? 'active' : 'normal';
  return <Progress percent={progress} status={status} strokeWidth={8} />;
};

export default ProgressBar;
