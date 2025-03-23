import { request } from '@umijs/max';

export async function queryAnalysisResult(fileId: number) {
  return request(`http://localhost:5000/api/v1/analysis/result/${fileId}`, {
    method: 'GET',
  });
} 