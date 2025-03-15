<template>
  <a-config-provider :theme="theme">
    <a-layout class="layout-container" :class="themeMode">
      <!-- 顶部导航栏 -->
      <a-layout-header class="header">
        <div class="header-content">
          <div class="left-section">
            <h1 class="system-title">MalTact 样本分析集成系统</h1>
            <a-menu
              v-model:selectedKeys="topMenuSelectedKeys"
              mode="horizontal"
              class="nav-menu"
              :theme="menuTheme"
            >
              <a-menu-item key="static">静态检测</a-menu-item>
              <a-menu-item key="model">模型检测</a-menu-item>
              <a-menu-item key="sandbox">沙箱检测</a-menu-item>
              <a-menu-item key="assembly">汇编分析</a-menu-item>
              <a-menu-item key="dynamic">动态模拟</a-menu-item>
            </a-menu>
          </div>
          <div class="right-section">
            <a-button shape="circle" @click="toggleTheme">
              <template #icon>
                <bulb-filled v-if="isDarkMode" />
                <bulb-outlined v-else />
              </template>
            </a-button>
            <a-dropdown>
              <a-menu class="user-menu" :theme="menuTheme">
                <a-menu-item key="user">
                  <a-avatar
                    :size="40"
                    src="https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png"
                  />
                </a-menu-item>
              </a-menu>
              <template #overlay>
                <a-menu @click="handleUserMenuClick">
                  <a-menu-item key="submit">提交记录</a-menu-item>
                  <a-menu-item key="profile">账号信息</a-menu-item>
                  <a-menu-item key="logout">退出登录</a-menu-item>
                </a-menu>
              </template>
            </a-dropdown>
          </div>
        </div>
      </a-layout-header>

      <!-- 主内容区 -->
      <a-layout class="main-content">
        <!-- 侧边栏 -->
        <a-layout-sider
          width="240"
          class="side-menu"
          v-if="showSidebar"
          :theme="menuTheme"
        >
          <a-menu
            v-model:selectedKeys="sideMenuSelectedKeys"
            mode="vertical"
            :theme="menuTheme"
          >
            <a-menu-item
              v-for="item in sideMenuItems"
              :key="item.key"
            >
              {{ item.label }}
            </a-menu-item>
          </a-menu>
        </a-layout-sider>

        <!-- 内容区域 -->
        <a-layout-content class="content-wrapper">
          <div class="content">
            <router-view />
          </div>
        </a-layout-content>
      </a-layout>

      <!-- 底部栏 -->
      <a-layout-footer class="footer">
        <div class="footer-content">
          © 2023 MalTact 样本分析系统. 作者：XXX技术团队
        </div>
      </a-layout-footer>
    </a-layout>
  </a-config-provider>
</template>

<script setup>
import { ref, computed } from 'vue';
import { BulbFilled, BulbOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';

// 主题切换逻辑
const isDarkMode = ref(localStorage.getItem('theme') === 'dark');
const themeMode = computed(() => isDarkMode.value ? 'dark-theme' : 'light-theme');
const menuTheme = computed(() => isDarkMode.value ? 'dark' : 'light');
const theme = computed(() => ({
  algorithm: isDarkMode.value ? theme.darkAlgorithm : theme.defaultAlgorithm,
}));

const toggleTheme = () => {
  isDarkMode.value = !isDarkMode.value;
  localStorage.setItem('theme', isDarkMode.value ? 'dark' : 'light');
};

// 用户菜单点击处理
const handleUserMenuClick = ({ key }) => {
  switch (key) {
    case 'submit':
      // 跳转到提交记录页面
      break;
    case 'profile':
      // 跳转到账号信息页面
      break;
    case 'logout':
      message.success('已退出登录');
      break;
  }
};

const topMenuSelectedKeys = ref(['static']);
const sideMenuSelectedKeys = ref(['1']);

// 侧边栏配置
const sideMenuConfig = {
  static: [
    { key: '1', label: '基础信息' },
    { key: '2', label: '规则匹配' },
    { key: '3', label: '字符串分析' }
  ],
  user: [
    { key: '4', label: '提交记录' },
    { key: '5', label: '账号信息' }
  ]
  // 其他菜单项的配置...
};

const showSidebar = computed(() => {
  return topMenuSelectedKeys.value[0] !== 'user'; // 示例逻辑
});

const sideMenuItems = computed(() => {
  const currentMenu = topMenuSelectedKeys.value[0];
  return sideMenuConfig[currentMenu] || [];
});
</script>

<style scoped>
.layout-container {
  min-height: 100vh;
}

.header {
  background: #fff;
  padding: 0 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 64px;
}

.left-section {
  display: flex;
  align-items: center;
}

.system-title {
  font-size: 18px;
  font-weight: 600;
  margin-right: 40px;
  color: #1890ff;
}

.nav-menu {
  line-height: 62px;
  border-bottom: none;
}

.user-menu {
  line-height: 62px;
}

.main-content {
  margin: 24px;
  min-height: calc(100vh - 112px);
}

.side-menu {
  background: #fff;
  margin-right: 24px;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.content-wrapper {
  padding: 24px;
  background: #fff;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.footer {
  text-align: center;
  background: #f0f2f5;
  padding: 24px 50px;
}

.footer-content {
  color: rgba(0,0,0,0.65);
}

.dark-theme {
  --bg-color: #1f1f1f;
  --text-color: rgba(255, 255, 255, 0.85);
  --border-color: #434343;
}

.light-theme {
  --bg-color: #ffffff;
  --text-color: rgba(0, 0, 0, 0.85);
  --border-color: #d9d9d9;
}

.layout-container {
  background: var(--bg-color);
  color: var(--text-color);
  transition: all 0.3s;
}

.header {
  background: var(--bg-color) !important;
  border-bottom: 1px solid var(--border-color);
}

.system-title {
  color: var(--text-color);
}

.side-menu,
.content-wrapper {
  background: var(--bg-color);
  border: 1px solid var(--border-color);
}

/* 调整原有样式 */
.right-section {
  display: flex;
  align-items: center;
  gap: 16px;
}

.user-menu {
  :deep(.ant-menu-item) {
    padding: 0 8px !important;
  }
}


/* 新增主题相关样式 */
.dark-theme {
  --bg-color: #1f1f1f;
  --text-color: rgba(255, 255, 255, 0.85);
  --border-color: #434343;
}

.light-theme {
  --bg-color: #ffffff;
  --text-color: rgba(0, 0, 0, 0.85);
  --border-color: #d9d9d9;
}

.layout-container {
  background: var(--bg-color);
  color: var(--text-color);
  transition: all 0.3s;
}

.header {
  background: var(--bg-color) !important;
  border-bottom: 1px solid var(--border-color);
}

.system-title {
  color: var(--text-color);
}

.side-menu,
.content-wrapper {
  background: var(--bg-color);
  border: 1px solid var(--border-color);
}

/* 调整原有样式 */
.right-section {
  display: flex;
  align-items: center;
  gap: 16px;
}

.user-menu {
  :deep(.ant-menu-item) {
    padding: 0 8px !important;
    &:hover {
      background: transparent !important;
    }
  }
}

/* 深色模式菜单覆盖 */
:deep(.ant-menu-dark) {
  background: var(--bg-color);

  .ant-menu-item {
    color: var(--text-color);

    &:hover {
      background: rgba(255, 255, 255, 0.1) !important;
    }
  }
}

:deep(.ant-menu-light) {
  background: var(--bg-color);

  .ant-menu-item {
    color: var(--text-color);

    &:hover {
      background: rgba(0, 0, 0, 0.06) !important;
    }
  }
}
</style>
