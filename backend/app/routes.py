from flask import jsonify, request
from app import app
from app.database import (
    init_db, get_all_samples, get_sample_by_id,
    insert_sample, insert_histogram_features,
    insert_pe_features, insert_engineered_features
)
import os
import json
import numpy as np
import hashlib
import random

# 初始化数据库
init_db()

@app.route('/api/samples', methods=['GET'])
def api_samples():
    """获取所有样本的列表"""
    samples = get_all_samples()
    return jsonify(samples)

@app.route('/api/samples/<int:sample_id>', methods=['GET'])
def api_sample_detail(sample_id):
    """获取特定样本的详细信息"""
    sample = get_sample_by_id(sample_id)
    if not sample:
        return jsonify({'error': 'Sample not found'}), 404
    return jsonify(sample)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """获取统计信息"""
    samples = get_all_samples()
    total_count = len(samples)
    malicious_count = sum(1 for s in samples if s['is_malicious'] == 1)
    benign_count = total_count - malicious_count
    
    return jsonify({
        'total_samples': total_count,
        'malicious_samples': malicious_count,
        'benign_samples': benign_count,
        'detection_rate': round(malicious_count / total_count * 100, 2) if total_count else 0,
        'Access-Control-Allow-Origin': 'http://localhost:8000',
        'Access-Control-Allow-Credentials': 'true'
    }) 
    
@app.route('/api/samples/upload', methods=['POST'])
def api_upload_sample():
    """上传样本文件进行分析"""
    # 在实际应用中，这里应该处理文件上传并提取特征
    # 为了演示，我们使用模拟数据
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 计算文件哈希
    file_content = file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    file_size = len(file_content)
    
    # 模拟分类结果
    is_malicious = random.choice([0, 1])
    classification_result = "恶意软件" if is_malicious else "正常软件"
    
    # 插入样本基本信息
    sample_id = insert_sample(
        file_hash=file_hash,
        file_name=file.filename,
        file_size=file_size,
        is_malicious=is_malicious,
        classification_result=classification_result
    )
    
    # 模拟提取特征并插入数据库
    # 1. 直方图特征
    byte_histogram = [random.randint(0, 10000) for _ in range(256)]
    entropy_histogram = [random.randint(0, 10000) for _ in range(256)]
    insert_histogram_features(sample_id, byte_histogram, entropy_histogram)
    
    # 2. PE静态特征
    general_info = {
        "debug_size": random.randint(0, 1000),
        "tls_size": random.randint(0, 1000),
        "relocations_size": random.randint(0, 1000),
        "major_version": random.randint(1, 10),
        "minor_version": random.randint(0, 10),
        "num_data_directories": random.randint(10, 16)
    }
    
    header_info = {
        "pe_signature": "PE",
        "machine_type": "I386",
        "timestamp": "2021-01-01",
        "num_sections": random.randint(3, 10),
        "pointer_symbol_table": random.randint(0, 1000000),
        "characteristics": ["EXECUTABLE", "32BIT"]
    }
    
    section_info = {
        "entry": ".text",
        "sections": [
            {
                "name": ".text",
                "size": random.randint(1000, 100000),
                "entropy": round(random.uniform(5.0, 8.0), 2),
                "vsize": random.randint(1000, 100000),
                "props": ["CNT_CODE", "MEM_EXECUTE", "MEM_READ"]
            },
            {
                "name": ".data",
                "size": random.randint(1000, 100000),
                "entropy": round(random.uniform(2.0, 6.0), 2),
                "vsize": random.randint(1000, 100000),
                "props": ["CNT_INITIALIZED_DATA", "MEM_READ", "MEM_WRITE"]
            }
        ]
    }
    
    exports_info = {
        "exports": [
            {"name": "Function1", "address": "0x1000"},
            {"name": "Function2", "address": "0x2000"}
        ]
    }
    
    insert_pe_features(sample_id, general_info, header_info, section_info, exports_info)
    
    # 3. 特征工程特征
    section_features = {
        "entry": 5,
        "size_R": random.randint(10000, 500000),
        "size_W": random.randint(10000, 500000),
        "size_X": random.randint(10000, 500000),
        "entr_R": round(random.uniform(5.0, 7.0), 2),
        "entr_W": round(random.uniform(2.0, 5.0), 2),
        "entr_X": round(random.uniform(6.0, 8.0), 2),
        "size_R_weight": random.randint(0, 100),
        "size_W_weight": random.randint(0, 100),
        "size_X_weight": random.randint(0, 100),
        "entr_R_weight": random.randint(0, 100),
        "entr_W_weight": random.randint(0, 100),
        "entr_X_weight": random.randint(0, 100),
        "rsrc_num": random.randint(0, 3),
        "section_num": random.randint(3, 10),
        "file_size": file_size
    }
    
    string_match = {
        "mz_count": random.randint(0, 10),
        "mz_mean": round(random.uniform(0, 30), 1),
        "pe_count": random.randint(0, 10),
        "pe_mean": round(random.uniform(0, 30), 1),
        "pool_count": random.randint(0, 10),
        "pool_mean": round(random.uniform(0, 30), 1),
        "cpu_count": random.randint(0, 10),
        "cpu_mean": round(random.uniform(0, 30), 1),
        "gpu_count": random.randint(0, 10),
        "gpu_mean": round(random.uniform(0, 30), 1),
        "coin_count": random.randint(0, 10),
        "coin_mean": round(random.uniform(0, 30), 1),
        "btc_count": random.randint(0, 10),
        "btc_mean": round(random.uniform(0, 30), 1),
        "ltc_count": random.randint(0, 10),
        "ltc_mean": round(random.uniform(0, 30), 1),
        "xmr_count": random.randint(0, 10),
        "xmr_mean": round(random.uniform(0, 30), 1),
        "paths_count": random.randint(0, 10),
        "paths_mean": round(random.uniform(0, 30), 1),
        "regs_count": random.randint(0, 10),
        "regs_mean": round(random.uniform(0, 30), 1),
        "urls_count": random.randint(0, 10),
        "urls_mean": round(random.uniform(0, 30), 1),
        "ips_count": random.randint(0, 10),
        "ips_mean": round(random.uniform(0, 30), 1)
    }
    
    yara_match = {
        "packer_count": random.randint(0, 3),
        "yargen_count": random.randint(0, 3)
    }
    
    string_count = {
        "av_count": random.randint(0, 10),
        "dbg_count": random.randint(0, 5),
        "pool_name_count": random.randint(0, 20),
        "algorithm_name_count": random.randint(0, 5),
        "coin_name_count": random.randint(0, 10)
    }
    
    opcode_features = {
        "opcode_min": random.randint(1, 10),
        "opcode_max": random.randint(500, 1000),
        "opcode_sum": random.randint(5000, 20000),
        "opcode_mean": round(random.uniform(40, 60), 2),
        "opcode_var": round(random.uniform(4000, 6000), 2),
        "opcode_count": random.randint(200, 400),
        "opcode_uniq": random.randint(100, 200)
    }
    
    insert_engineered_features(
        sample_id, section_features, string_match,
        yara_match, string_count, opcode_features
    )
    
    return jsonify({
        'success': True,
        'sample_id': sample_id,
        'message': '已成功上传并分析样本'
    })

