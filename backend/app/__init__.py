from flask import Flask
from flask_cors import CORS

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from app.config import Config


# 创建数据库实例
db = SQLAlchemy()
migrate = Migrate()

def create_app():

    # 注册 Flask 应用
    app = Flask(__name__)
    CORS(app)

    # 初始化配置
    Config.init_folders()
    app.config.from_object(Config)

    # 初始化数据库和迁移
    from .models.model import UploadHistory, BasicInfo, PEInfo, YaraMatch, SigmaMatch, AnalyzeStrings
    db.init_app(app)
    migrate.init_app(app, db)

    # 注册蓝图
    from app.api.v1 import api_v1 as api_v1_bp
    from app.api.v2 import api_v2 as api_v2_bp
    # Blueprint 实现不同版本 API 模块化
    app.register_blueprint(api_v1_bp, url_prefix='/api/v1')
    app.register_blueprint(api_v2_bp, url_prefix='/api/v2')
    

    return app