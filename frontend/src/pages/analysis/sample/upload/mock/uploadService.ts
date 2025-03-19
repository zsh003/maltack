import { mockAnalysisResponse } from './mockData';

// 模拟上传函数
export const mockUpload = (file: File): Promise<any> => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (Math.random() > 0.1) { // 模拟90%的成功率
        resolve(mockAnalysisResponse);
      } else {
        reject({ message: '模拟上传失败' });
      }
    }, 2000); // 模拟2秒的上传时间
  });
};
