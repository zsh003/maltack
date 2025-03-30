from flask import jsonify, Blueprint, request
from flask import Flask
import os
from datetime import datetime
import json

from .controllers import get_hello_v1
from app.services.analysis import analyze_file
from app.config import Config
from app import db
from app.models.model import UploadHistory

from app.api.v1.features import (
    save_byte_histogram, save_byte_entropy, save_pe_static_feature,
    save_feature_engineering
)
from app.utils.feature_extraction import (
    extract_byte_histogram, extract_byte_entropy,
    extract_pe_static_features, extract_feature_engineering
)
# from app.services.upload_service import handle_file_upload

app = Flask(__name__)

api_v1 = Blueprint('api_v1_bp', __name__)

@api_v1.route('/hello')
def hello_v1():
    return jsonify(get_hello_v1())

@api_v1.route('/login', methods=['POST'])
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

@api_v1.route('/userinfo', methods=['GET'])
def userinfo():
    # 这里可以添加 JWT 验证逻辑
    user_info = {'name': 'John Doe', 'email': 'john.doe@example.com'}
    return jsonify(user_info)

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


@api_v1.route('/upload', methods=['POST'])
def upload_file():
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
            threat_level="unknown",
            status="analyzing"
        )
        db.session.add(new_upload)
        db.session.commit()
        app.logger.debug('DB saved successfully.')

        new_upload.status = "analyzing"
        result = analyze_file(filepath)
        app.logger.debug('File analyzed successfully.')

        # 保存分析结果到数据库
        save_analysis_results(new_upload.file_id, result)
        new_upload.status = "completed"
        db.session.commit()

    except Exception as e:
        app.logger.error(f'Analysis failed: {str(e)}')
        new_upload.status = "failed"
        db.session.commit()
    

    # 第二部分，提取特征
    try: 
        # 提取特征
        with open(filepath, 'rb') as f:
            file_content = f.read()
            
            # 提取字节直方图特征
            histogram_data = extract_byte_histogram(file_content)
            save_byte_histogram(new_upload.file_id, histogram_data)
            app.logger.debug('histogram saved successfully')
            
            # 提取字节熵特征
            entropy_data = extract_byte_entropy(file_content)
            save_byte_entropy(new_upload.file_id, entropy_data)
            app.logger.debug('entropy saved successfully')
            
            # 提取PE静态特征
            pe_features = extract_pe_static_features(filepath)
            for feature_type, feature_data in pe_features.items():
                save_pe_static_feature(new_upload.file_id, feature_type, feature_data)
            app.logger.debug('pe static feature saved successfully')
            
            # 提取特征工程数据
            section_info, string_matches, yara_matches, opcode_features, boolean_features = \
                extract_feature_engineering(filepath)
            save_feature_engineering(
                new_upload.file_id,
                section_info,
                string_matches,
                yara_matches,
                opcode_features,
                boolean_features
            )
            app.logger.debug('feature engineering saved successfully')
        
        # 更新上传记录状态
        new_upload.status = 'completed'
        db.session.commit()
        
        return jsonify({
            'message': '文件上传成功',
            'file_id': new_upload.file_id
        })
    
    except Exception as e:
        if 'new_upload' in locals():
            new_upload.status = 'failed'
            new_upload.error_message = str(e)
            db.session.commit()
        return jsonify({'error': str(e)}), 500

# 保存分析结果到数据库
def save_analysis_results(file_id, result):
    from app.models.model import (BasicInfo, PEInfo, YaraMatch, 
                                SigmaMatch, AnalyzeStrings)

    # 保存基础信息
    basic_info = BasicInfo(
        file_id=file_id,
        file_name=result['basic_info']['file_name'],
        file_size=result['basic_info']['file_size'],
        file_type=result['basic_info']['file_type'],
        mime_type=result['basic_info']['mime_type'],
        analyze_time=result['basic_info']['analyze_time'],
        md5=result['basic_info']['md5'],
        sha1=result['basic_info']['sha1'],
        sha256=result['basic_info']['sha256']
    )
    db.session.add(basic_info)

    app.logger.debug('Analysis result saved in basic_info successfully.')

    # 保存PE信息（如果存在）
    if 'pe_info' in result and not isinstance(result['pe_info'], dict):
        pe_info = PEInfo(
            file_id=file_id,
            machine_type=result['pe_info'].get('machine_type'),
            timestamp=result['pe_info'].get('timestamp'),
            subsystem=result['pe_info'].get('subsystem'),
            dll_characteristics=result['pe_info'].get('dll_characteristics'),
            sections=json.dumps(result['pe_info'].get('sections', [])),
            imports=json.dumps(result['pe_info'].get('imports', [])),
            exports=json.dumps(result['pe_info'].get('exports', []))
        )
    else:
        pe_info = PEInfo(
            file_id=file_id,
            machine_type=None,
            timestamp=None,
            subsystem=None,
            dll_characteristics=None,
            sections=None,
            imports=None,
            exports=None
        )
    db.session.add(pe_info)
    app.logger.debug('Analysis result saved in pe_info successfully.')

    # 保存YARA匹配结果
    if 'yara_match' in result:
        yara_match = YaraMatch(
            file_id=file_id,
            rule_name=result['yara_match']['rule_name'],
            tags=result['yara_match']['tags'],
            strings=json.dumps(result['yara_match']),
            meta=result['yara_match']['meta']
        )
    else:
        yara_match = YaraMatch(
            file_id=file_id,
            rule_name=None,
            tags=None,
            strings=None,
            meta=None
        )
    db.session.add(yara_match) 
    app.logger.debug('Analysis result saved in yara_match successfully.')

    # 保存Sigma匹配结果
    if 'sigma_matches' in result:
        sigma_match = SigmaMatch(
            file_id=file_id,
            matches=json.dumps(result['sigma_matches'])
        )
    else:
        sigma_match = SigmaMatch(
            file_id=file_id,
            matches=None
        )
    db.session.add(sigma_match)
    app.logger.debug('Analysis result saved in sigma_match successfully.')

    # 保存字符串分析结果
    if 'string_info' in result:
        analyze_strings = AnalyzeStrings(
            file_id=file_id,
            ascii_strings=json.dumps(result['string_info']['ascii_strings']),
            unicode_strings=json.dumps(result['string_info']['unicode_strings'])
        )
    else:
        analyze_strings = AnalyzeStrings(
            file_id=file_id,
            ascii_strings=None,
            unicode_strings=None
        )
    db.session.add(analyze_strings)
    app.logger.debug('Analysis result saved in analyze_strings successfully.')

    db.session.commit()

    app.logger.debug('分析结果保存成功')

@api_v1.route('/analysis/result/overview/<int:file_id>', methods=['GET'])
def get_analysis_result(file_id):
    upload = UploadHistory.query.get(file_id)
    if not upload:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify({
        'basic_info': upload.basic_info.as_dict() if upload.basic_info else None,
        'pe_info': upload.pe_info.as_dict() if upload.pe_info else None,
        'yara_matches': upload.yara_match.as_dict() if upload.yara_match else None,
        'sigma_matches': upload.sigma_match.as_dict() if upload.sigma_match else None,
        'string_info': upload.analyze_strings.as_dict() if upload.analyze_strings else None
    })

@api_v1.route('/analysis/result/basic-info/<int:file_id>', methods=['GET'])
def get_basic_info(file_id):
    upload = UploadHistory.query.get(file_id)
    if not upload:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify({
        'basic_info': upload.basic_info.as_dict() if upload.basic_info else None
    })

@api_v1.route('/analysis/result/pe-info/<int:file_id>', methods=['GET'])
def get_pe_info(file_id):
    upload = UploadHistory.query.get(file_id)
    if not upload:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify({
        'pe_info': upload.pe_info.as_dict() if upload.pe_info else None
    })

@api_v1.route('/analysis/result/yara-rules/<int:file_id>', methods=['GET'])
def get_yara_rules(file_id):
    upload = UploadHistory.query.get(file_id)
    if not upload:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify({
        'yara_matches': upload.yara_match.as_dict() if upload.yara_match else None
    })

@api_v1.route('/analysis/result/sigma-rules/<int:file_id>', methods=['GET'])
def get_sigma_rules(file_id):
    upload = UploadHistory.query.get(file_id)
    if not upload:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify({
        'sigma_matches': upload.sigma_match.as_dict() if upload.sigma_match else None
    })

@api_v1.route('/analysis/result/strings/<int:file_id>', methods=['GET'])
def get_strings(file_id):
    upload = UploadHistory.query.get(file_id)
    if not upload:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify({
        'string_info': upload.analyze_strings.as_dict() if upload.analyze_strings else None
    })