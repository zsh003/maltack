### backend

- pip环境：
```bash
cd backend
pip install -r requirements.txt # python version: 3.11.7
```

- 数据库初始化
```bash
flask db init   # 仅第一次运行
```

- 数据库迁移更改

每当修改了模型（如新增或删除表），你需要创建一个迁移脚本，运行迁移脚本来更新数据库到最新版本：
```bash
flask db migrate -m "Initial migration"
flask db upgrade
```

回滚到前一版本的数据库结构，用于撤销最近的迁移操作
```bash
flask db downgrade
```


### frontend

```bash
tyarn # 安装依赖
npm run serve # 开发模式运行
npm run build # 编译项目
```
