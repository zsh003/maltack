import { Tabs } from 'antd';
import { useLocation, useNavigate } from '@umijs/max';
import HistogramFeatures from './histogram';
import PEStaticFeatures from './pe-static-feature';
import FeatureEngineering from './feature-engineering';

const { TabPane } = Tabs;

const MachineLearningFeatures: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const pathname = location.pathname;

  const getTabKey = () => {
    if (pathname.includes('/histogram')) return 'histogram';
    if (pathname.includes('/pe-static-feature')) return 'pe-static-feature';
    if (pathname.includes('/feature-engineering')) return 'feature-engineering';
    return 'histogram';
  };

  const handleTabChange = (key: string) => {
    const basePath = '/analysis/machine-learning';
    switch (key) {
      case 'histogram':
        navigate(`${basePath}/histogram`);
        break;
      case 'pe-static-feature':
        navigate(`${basePath}/pe-static-feature`);
        break;
      case 'feature-engineering':
        navigate(`${basePath}/feature-engineering`);
        break;
      default:
        navigate(`${basePath}/histogram`);
    }
  };

  return (
    <div>
      <Tabs activeKey={getTabKey()} onChange={handleTabChange}>
        <TabPane tab="直方图特征" key="histogram">
          <HistogramFeatures />
        </TabPane>
        <TabPane tab="PE静态特征" key="pe-static-feature">
          <PEStaticFeatures />
        </TabPane>
        <TabPane tab="特征工程" key="feature-engineering">
          <FeatureEngineering />
        </TabPane>
      </Tabs>
    </div>
  );
};

export default MachineLearningFeatures;
