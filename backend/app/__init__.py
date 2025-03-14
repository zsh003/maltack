from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.routes.main import main_bp
from app.routes.api import api_bp

def create_app():

    # 注册 Flask 应用
    app = Flask(__name__)
    
    # 加载配置
    Config.init_folders()

    # 注册main服务
    app.register_blueprint(main_bp)

    # 注册api服务
    CORS(api_bp, supports_credentials=True)
    app.register_blueprint(api_bp, url_prefix='/api/v1.0')

    return app