import os

class Config:
    DEBUG = True
    TESTING = False
    SECRET_KEY = 'your_secret_key'

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