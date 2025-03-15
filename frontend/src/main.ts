import { createApp } from 'vue'
import App from './App.vue'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'
import router from './router'
//import store from './stores';

//createApp(App).use(store).use(router).use(Antd).mount('#app');
createApp(App).use(router).use(Antd).mount('#app')
