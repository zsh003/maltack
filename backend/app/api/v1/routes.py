from flask import jsonify, Blueprint
from .controllers import get_hello_v1
from app.services.analysis import analyze_file
import os

api_v1 = Blueprint('api_v1_bp', __name__)

@api_v1.route('/hello')
def hello_v1():
    return jsonify(get_hello_v1())

from flask import Blueprint, request, jsonify


@api_v1.route('/analyze', methods=['POST'])
def api_analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 保存文件
    filepath = os.path.join(os.getcwd(), 'uploads', file.filename)
    file.save(filepath)
    
    # 分析文件
    result = analyze_file(filepath)
    
    return jsonify(result)
