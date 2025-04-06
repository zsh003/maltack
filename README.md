# 恶意PE软件特征检测与识别系统

本项目是一个基于集成学习的恶意PE软件特征检测与识别系统，提供了PE文件的特征提取、分析和可视化功能。

## 项目结构

```
mal_ana/
├── backend/          # Flask后端API服务
│   ├── app/          # 应用代码
│   ├── data/         # 数据存储
│   └── run.py        # 启动脚本
└── frontend/         # React前端项目
    ├── src/          # 源代码
    │   ├── components/  # 组件
    │   ├── pages/    # 页面
    │   ├── services/ # API服务
    │   └── utils/    # 工具函数
    └── .umirc.ts     # UMI配置
```

## 功能特性

1. **特征提取与分析**
   - 直方图特征：字节分布直方图和字节熵直方图
   - PE静态特征：PE头信息、节区属性等
   - 特征工程：Yara规则匹配、字符串匹配、操作码分析等

2. **数据可视化**
   - 交互式直方图展示
   - PE结构可视化
   - 特征雷达图

3. **样本管理**
   - 样本上传与自动分析
   - 样本列表与搜索
   - 详细的特征分析报告

## 安装与部署

### 后端

1. 进入backend目录
   ```
   cd backend
   ```

2. 安装依赖
   ```
   pip install -r requirements.txt
   ```

3. 启动服务
   ```
   python run.py
   ```
   
服务将在 http://localhost:5000 运行。

### 前端

1. 进入frontend目录
   ```
   cd frontend
   ```

2. 安装依赖
   ```
   npm install
   ```

3. 启动开发服务器
   ```
   npm start
   ```

前端将在 http://localhost:8000 运行。

## API文档

### 获取样本列表
```
GET /api/samples
```

### 获取样本详情
```
GET /api/samples/{id}
```

### 上传样本
```
POST /api/samples/upload
```

### 获取统计信息
```
GET /api/stats
```

## 技术栈

- 后端: Flask, SQLite, NumPy, Pandas
- 前端: React, UmiJS, Ant Design, ECharts

## 许可证

本项目采用 MIT 许可证
