from flask import Blueprint, request, jsonify
from app.services.analysis import analyze_file

api_bp = Blueprint('api', __name__)

@api_bp.route('/analyze', methods=['POST'])
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



