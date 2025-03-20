import os
import logging

# 数据库配置
HOST = '127.0.0.1'
PORT = 3306
DATABASE = 'maltack'
USERNAME = 'maltack'
PASSWORD = '123456'

class Config:
    DEBUG = True
    TESTING = False

    # 数据库配置
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?charset=utf8mb4'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    # # 密钥设置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mysecretkey') or 'your_secret_key'

    # 日志配置
    LOGGER_NAME = 'app'
    LOGGER_LEVEL = logging.DEBUG
    LOGGER_FILENAME = 'app.log'

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'uploads')
    RULES_FOLDER = os.path.join(BASE_DIR, '..', 'rules')
    YARA_RULES_PATH = os.path.join(RULES_FOLDER, 'rules.yar')
    SIGMA_RULES_PATH = os.path.join(RULES_FOLDER, 'rules.yml')

    @staticmethod
    def init_folders():
        for folder in [Config.UPLOAD_FOLDER, Config.RULES_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True

class ProductionConfig(Config):
    pass

if __name__ == "__main__":
    print(1)
    print(Config.BASE_DIR)
    print(Config.UPLOAD_FOLDER)