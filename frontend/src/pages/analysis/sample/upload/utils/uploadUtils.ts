import { message } from 'antd';

export const beforeUpload = (file: File, setIsAnalyzing: (value: boolean) => void, setUploadError: (value: string | null) => void, setUploadLoading: (value: boolean) => void): boolean => {
  setIsAnalyzing(true); // 开始分析
  setUploadError(null); // 清空错误
  const isValidSize = file.size / 1024 / 1024 < 100;
  if (!isValidSize) {
    message.error('文件大小不能超过100MB');
    return false;
  }
  setUploadLoading(true); // 设置加载状态
  return true; // 必须返回true才会继续上传
};

export const handleUploadSuccess = (response: any, setIsAnalyzing: (value: boolean) => void, setUploadLoading: (value: boolean) => void, setUploadError: (value: string | null) => void) => {
  setIsAnalyzing(false);
  setUploadLoading(false); // 关闭加载状态
  if (response && response.success) { // 假设后端返回成功标志
    message.success('文件分析完成');
  } else {
    setUploadError('分析失败，请检查文件格式');
    message.error('分析失败，请检查文件格式');
  }
};

export const handleUploadError = ({ file,}: { file: any; }, setIsAnalyzing: (value: boolean) => void, setUploadLoading: (value: boolean) => void, setUploadError: (value: string | null) => void) => {
  setIsAnalyzing(false);
  setUploadLoading(false);
  const errorMessage = `上传失败：${file.response?.message || '未知错误'}`;
  setUploadError(errorMessage);
  message.error(errorMessage);
};

export const handleUploadChange = (info: any) => {
  if (info.file.status === 'done') {
    message.success(`${info.file.name} 文件上传成功`);
  } else if (info.file.status === 'error') {
    message.error(`${info.file.name} 文件上传失败`);
  }
};
