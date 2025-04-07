import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NODE_ENV === 'development' ? 'http://localhost:5000/api' : '/api',
  timeout: 60000,
  withCredentials: true
});

export async function fetchSamples() {
  const response = await api.get('/samples');
  return response.data;
}

export async function fetchSampleDetail(id: number) {
  const response = await api.get(`/samples/${id}`);
  return response.data;
}

export async function uploadSample(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },
  });
  
  return response.data;
}

export async function fetchStats() {
  const response = await api.get('/stats');
  return response.data;
}

export default api; 