from app import create_app
from dotenv import load_dotenv

'''
dotenv管理环境变量，load加载.env文件
FLASK_APP环境变量：指向你的应用入口文件
FLASK_DEBUG环境变量：启用调试模式，等同于python run.py方法中，下面app.run(debug=True)
FLASK_RUN_HOST, FLASK_RUN_PORT：指定主机和端口
'''

'''
flask run  # 启动环境
'''
if __name__ == '__main__':

    load_dotenv(dotenv_path='./.env')
    app = create_app()

    app.run() 