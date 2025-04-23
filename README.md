<h1 align="center">✨ Maltack - Elaborate Detection for PE ✨</h1>
<p align="center">
  一个集成了PE的各种特征提取、分析、可视化功能的系统
  <br><br>
  <a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/github/license/zsh003/maltack"></a>
  <a href="https://github.com/zsh003/maltack"><img src="https://img.shields.io/github/languages/top/zsh003/maltack"></a>
  <a href="https://github.com/zsh003/maltack/issues"><img src="https://img.shields.io/github/issues/zsh003/maltack"></a>
  <br>
  <img alt="react Version" src="https://img.shields.io/badge/react-v17.0-8A2BE2">	   <img alt="umi Version" src="https://img.shields.io/badge/umi-v3.5.20-8A2BE2">   <img alt="ant Version" src="https://img.shields.io/badge/antd-v4.16.13-8A2BE2">
  <img alt="ts Version" src="https://img.shields.io/badge/typescript-v4.1.2-8A2BE2">
  <br>
  <img alt="fastapi Version" src="https://img.shields.io/badge/fastapi-v0.115.12-blue">
  <img alt="sqlite Version" src="https://img.shields.io/badge/sqlite3-v2.0.4-blue">



## 项目结构

```
mal_ana/
├── backend/          # FastAPI后端服务
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
   - PE特征饼状图
   - 特征雷达图

3. **样本管理**
   - 样本上传与自动分析
   - 样本列表与搜索
   - 详细的特征分析

## 技术栈

- 后端: FastAPI, SQLite, NumPy, Pandas
- 前端: React, UmiJS, Ant Design, ECharts

## TODO List
- [ ] 添加更多的PE元数据分析
- [ ] 完善PE节区部分提取
- [ ] 完善特征工程部分功能实现
- [ ] 添加机器学习检测部分
- [ ] 添加集成学习和自定义数据集学习部分
- [ ] 添加多类型PE检测
- [ ] 添加Docker封装
- [ ] 添加AI接口
- [ ] 改善前端菜单页面
- [ ] 添加PE结构可视化
- [ ] 添加脱壳功能
- [ ] 完善Yara规则匹配
- [ ] 添加十六进制编辑功能
- [ ] 添加反汇编功能




## 手动安装与部署

1. 克隆仓库：
```bash
git clone https://github.com/zsh003/maltack.git
cd maltack
```

2. 创建后端环境
```bash
# Python version: 3.11.x
cd backend
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
python create_mock_data.py # 初始化数据库 (init database)
python run.py # 启动后端服务 (start server)
```

3. 创建前端环境
```bash
# Node.js version >= 18.x
cd frontend
npm install
npm start
```

4. 访问系统
- 前端页面: http://localhost:8000
- API接口文档: http://localhost:5000/docs#/
- 数据存储: `backend/data/malware_features.db` 




## 许可证

本项目采用 Apache 2.0 许可证
