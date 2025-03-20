import axios from 'axios';

export const handleFileUpload = async (file: File) => {
  try {
    const response = await axios.post('/v1/upload', file, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    throw new Error('上传失败');
  }
};
