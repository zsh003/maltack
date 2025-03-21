// apiService.ts
import axios from 'axios';

export const handleFileUpload = async (file: File, onProgress: (percent: number) => void) => {
  try {
    const config = {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent: ProgressEvent) => {
        const percent = Math.round((progressEvent.loaded / progressEvent.total!) * 100);
        onProgress(percent); // 触发进度回调
      },
    };
    const response = await axios.post('http://localhost:5000/api/v1/upload', file, config);
    return response.data;
  } catch (error) {
    throw new Error('上传失败');
  }
};
