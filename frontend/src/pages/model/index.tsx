import React from 'react';
import { Card, Typography, Spin } from 'antd';
import { useRequest } from 'umi';
import ModelAnalysisCard from '@/components/ModelAnalysisCard';
import { getModelAnalysis } from '@/services/api';

const { Title, Paragraph } = Typography;

const ModelAnalysisPage: React.FC = () => {
  const { data: modelData, loading } = useRequest(getModelAnalysis);

  if (loading) {
    return <Card><Spin /></Card>;
  }

  return (
    <div className="model-analysis-page" style={{ padding: '24px' }}>
      <Title level={2}>机器学习模型分析</Title>
      
      <Paragraph>
        本系统采用多层集成学习架构，包含三个主要特征提取和分析模块：
      </Paragraph>
      
      <Paragraph>
        <ul>
          <li>
            <strong>直方图特征CNN模型</strong>：
            使用字节直方图（256维）和字节熵直方图（256维）作为输入，
            通过卷积神经网络提取高级特征，有效识别加密/混淆等异常模式。
          </li>
          <li>
            <strong>PE静态特征集成模型</strong>：
            基于9个异构基模型（逻辑回归、XGBoost、随机森林等）的投票结果，
            分析PE文件的967维结构化特征。
          </li>
          <li>
            <strong>特征工程LightGBM模型</strong>：
            处理高级特征工程结果，包括节区分析、字符串匹配、YARA规则等，
            通过梯度提升决策树进行分类。
          </li>
        </ul>
      </Paragraph>

      <Paragraph>
        最终通过加权投票的方式融合三个模型的预测结果，充分利用不同特征的优势，
        提高整体检测准确率。下面展示了各个模型的性能指标和训练过程：
      </Paragraph>

      <ModelAnalysisCard data={modelData} />
    </div>
  );
};

export default ModelAnalysisPage; 