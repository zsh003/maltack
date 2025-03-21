from flask import jsonify, Blueprint, request
from flask import Flask
import os
from datetime import datetime

from .controllers import get_hello_v1
from app.services.analysis import analyze_file
from app.config import Config
from app import db
from app.models.model import UploadHistory
# from app.services.upload_service import handle_file_upload

app = Flask(__name__)

api_v1 = Blueprint('api_v1_bp', __name__)

@api_v1.route('/hello')
def hello_v1():
    return jsonify(get_hello_v1())

@app.route('/login', methods=['POST'])
def login():
    from app.models.user import User

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    with app.app_context():
        # 检查是否已经有用户
        if not User.query.first():
            admin_user = User(username='admin', password='123')
            db.session.add(admin_user)
            db.session.commit()

    user = User.query.filter_by(username=username, password=password).first()

    if user:
        return jsonify({'status': 'ok', 'token': 'your-jwt-token-here'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

@app.route('/userinfo', methods=['GET'])
def userinfo():
    # 这里可以添加 JWT 验证逻辑
    user_info = {'name': 'John Doe', 'email': 'john.doe@example.com'}
    return jsonify(user_info)


@api_v1.route('/upload', methods=['POST'])
def api_analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    try:
        # 保存文件并分析
        filepath = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        app.logger.debug('File saved successfully.')

        # 处理文件上传并保存数据库记录
        file_name = file.filename
        file_type = file.content_type
        new_upload = UploadHistory(
            file_url="http://localhost:5000/upload/" + file_name,
            file_name=file_name,
            file_type=file_type,
            environment="Windows 10",
            upload_time=datetime.utcnow(),
            threat_level="-",
            status="analyzing"
        )
        db.session.add(new_upload)
        db.session.commit()
        app.logger.debug('DB saved successfully.')

        result = analyze_file(filepath)
        app.logger.debug('File analyzed successfully.')

        return jsonify({'success': True, 'data': result, "message": "File uploaded successfully", "file_id": new_upload.file_id} )
    except Exception as e:
        app.logger.debug(str(e))
        return jsonify({'success': False, 'error': str(e)})
    
    return jsonify(result)


@api_v1.route('/upload_history', methods=['GET'])
def get_upload_history():
    uploads = UploadHistory.query.all()
    history = [{
        "file_name": upload.file_name,
        "file_type": upload.file_type,
        "environment": upload.environment,
        "upload_time": upload.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
        "threat_level": upload.threat_level,
        "status": upload.status,
    } for upload in uploads]
    return jsonify({"upload_history": history}), 200