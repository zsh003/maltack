// router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import UploadView from '../views/UploadView.vue'
import AnalysisView from '../views/AnalysisView.vue'

const routes = [
  // {
  //   path: '/',
  //   redirect: '/',
  // },
  {
    path: '/upload',
    name: 'Upload',
    component: UploadView,
  },
  {
    path: '/analysis',
    name: 'Analysis',
    component: AnalysisView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
