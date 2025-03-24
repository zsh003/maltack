import { useState, useCallback, useEffect } from 'react';

// 自定义hook，用于管理当前分析的文件ID
export default function useAnalysisModel() {
  const [currentFileId, setCurrentFileId] = useState<number>(() => {
    // 从本地存储初始化
    const saved = localStorage.getItem('currentFileId');
    return saved ? parseInt(saved) : 1;
  });

  useEffect(() => {
    // 同步到本地存储
    localStorage.setItem('currentFileId', currentFileId.toString());
  }, [currentFileId]);

  const updateCurrentFileId = useCallback((fileId: number) => {
    setCurrentFileId(fileId);
  }, []);

  return {
    currentFileId,
    updateCurrentFileId,
  };
}