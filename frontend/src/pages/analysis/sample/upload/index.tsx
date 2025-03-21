import React, { useEffect, useState } from 'react';
import FileUploader from './components/FileUploader';
import UploadHistory from './components/UploadHistory';
import axios from 'axios';

const Home = () => {
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const { data } = await axios.get('http://localhost:5000/api/v1/upload_history');
        setHistory(data.upload_history);
      } catch (error) {
        console.error('获取上传历史失败', error);
      }
    };

    fetchHistory();
  }, []);

  return (
      <div className="home-page">
        <FileUploader />
        <UploadHistory history={history} />
      </div>
  );
};

export default Home;
