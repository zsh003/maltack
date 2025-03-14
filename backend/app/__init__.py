from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.routes.main import main_bp
from app.routes.api import api_bp

def create_app():

    # ע�� Flask Ӧ��
    app = Flask(__name__)
    
    # ��������
    Config.init_folders()

    # ע��main����
    app.register_blueprint(main_bp)

    # ע��api����
    CORS(api_bp, supports_credentials=True)
    app.register_blueprint(api_bp, url_prefix='/api/v1.0')

    return app