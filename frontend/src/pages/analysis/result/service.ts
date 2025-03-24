import { request } from '@umijs/max';
import useAnalysisModel from '@/models/analysis';

export async function getAnalysisResult() {
  const { currentFileId } = useAnalysisModel();
  return request(`http://localhost:5000/api/v1/analysis/result/overview/${currentFileId}`, {
    method: 'GET',
  });
}

export async function getBasicInfo() {
  const { currentFileId } = useAnalysisModel();
  return request(`http://localhost:5000/api/v1/analysis/result/basic-info/${currentFileId}`, {
    method: 'GET',
  });
}

export async function getPEInfo() {
  const { currentFileId } = useAnalysisModel();
  return request(`http://localhost:5000/api/v1/analysis/result/pe-info/${currentFileId}`, {
    method: 'GET',
  });
}

export async function getYaraRules() {
  const { currentFileId } = useAnalysisModel();
  return request(`http://localhost:5000/api/v1/analysis/result/yara-rules/${currentFileId}`, {
    method: 'GET',
  });
}

export async function getSigmaRules() {
  const { currentFileId } = useAnalysisModel();
  return request(`http://localhost:5000/api/v1/analysis/result/sigma-rules/${currentFileId}`, {
    method: 'GET',
  });
}

export async function getStrings() {
  const { currentFileId } = useAnalysisModel();
  return request(`http://localhost:5000/api/v1/analysis/result/strings/${currentFileId}`, {
    method: 'GET',
  });
}
