import { defineConfig } from 'umi';

export default defineConfig({
  nodeModulesTransform: {
    type: 'none',
  },
  routes: [
    { path: '/', component: '@/pages/index' },
    { path: '/dashboard', component: '@/pages/dashboard/index' },
    { path: '/samples', component: '@/pages/samples/index' },
    { path: '/samples/:id', component: '@/pages/samples/detail' },
    { path: '/upload', component: '@/pages/upload/index' },
  ],
  layout: {
    name: '恶意PE软件特征检测与识别',
    locale: false,
  },
  fastRefresh: {},
  proxy: {
    '/api': {
      'target': 'http://localhost:5000',
      'changeOrigin': true,
    },
  },
}); 