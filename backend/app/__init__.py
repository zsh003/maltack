from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

from app import routes  # 导入路由 