import { request } from '@umijs/max';
import useAnalysisModel from '@/models/analysis';
import axios from "axios";

export async function getAnalysisResult(currentFileId: number) {
  const response = await axios.get(`http://localhost:5000/api/v1/analysis/result/overview/${currentFileId}`);
  return response.data;
}

export async function getBasicInfo(currentFileId: number) {
  const response = await request(`http://localhost:5000/api/v1/analysis/result/basic-info/${currentFileId}`);
  return response.data;
}

export async function getPEInfo(currentFileId: number) {
  return request(`http://localhost:5000/api/v1/analysis/result/pe-info/${currentFileId}`, {
    method: 'GET',
  });
}

export async function getYaraRules(currentFileId: number) {
  return request(`http://localhost:5000/api/v1/analysis/result/yara-rules/${currentFileId}`, {
    method: 'GET',
  });
}

export async function getSigmaRules(currentFileId: number) {
  return request(`http://localhost:5000/api/v1/analysis/result/sigma-rules/${currentFileId}`, {
    method: 'GET',
  });
}

export async function getStrings(currentFileId: number) {
  return request(`http://localhost:5000/api/v1/analysis/result/strings/${currentFileId}`, {
    method: 'GET',
  });
}
