from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.api.v1 import api_v1 as api_v1_bp
from app.api.v2 import api_v2 as api_v2_bp

def create_app():

    # 注册 Flask 应用
    app = Flask(__name__)
    
    # 加载配置
    Config.init_folders()
    # app.config.from_object('app.config.Config')

    # 注册蓝图
    # Blueprint 实现不同版本 API 模块化
    CORS(api_v1_bp, supports_credentials=True)
    CORS(api_v2_bp, supports_credentials=True)
    app.register_blueprint(api_v1_bp, url_prefix='/api/v1')
    app.register_blueprint(api_v2_bp, url_prefix='/api/v2')
    

    return app