from app import create_app
from dotenv import load_dotenv
import sys
from pathlib import Path

# 将项目根目录（backend）添加到 Python 路径
sys.path.append(str(Path(__file__).parent))

'''
dotenv管理环境变量，load加载.env文件
FLASK_APP环境变量：指向你的应用入口文件
FLASK_DEBUG环境变量：启用调试模式，等同于python run.py方法中，下面app.run(debug=True)
FLASK_RUN_HOST, FLASK_RUN_PORT：指定主机和端口
'''

'''
flask db init # 初始化迁移目录（仅第一次）
flask db migrate -m "Initial migration"  # 生成迁移脚本，首次运行时直接执行
flask db upgrade  # 执行迁移脚本，创建数据库结构
flask run  # 启动环境
'''

'''
每当修改了模型（如新增或删除表），你需要创建一个迁移脚本，运行迁移脚本来更新数据库：
flask db migrate -m "Update message"  # 生成迁移脚本
flask db upgrade  # 更新数据库
flask db downgrade  # 回滚上一版本
'''

if __name__ == '__main__':

    load_dotenv(dotenv_path='./.env')
    app = create_app()

    app.run() 