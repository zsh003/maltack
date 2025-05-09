from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .database import (
    get_all_samples, get_sample_by_id,
    insert_sample, insert_histogram_features, insert_pe_features,
    insert_engineered_features, insert_lief_features
)
from app.mock.lief_mock import generate_mock_lief_data
from .feature_extract.raw_features import ByteHistogram, ByteEntropyHistogram
from .feature_extract.lief_features import LiefFeatureExtractor
import os
import hashlib
from typing import List, Dict, Any
import random
import numpy as np
import pickle

app = FastAPI(title="恶意PE软件特征检测与识别系统")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/api/samples", response_model=List[Dict[str, Any]])
async def get_samples():
    """获取所有样本列表"""
    return get_all_samples()

@app.get("/api/samples/{sample_id}", response_model=Dict[str, Any])
async def get_sample_detail(sample_id: int):
    """获取样本详细信息"""
    sample = get_sample_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample

@app.get("/api/stats", response_model=Dict[str, Any])
async def get_stats():
    """获取统计信息"""
    samples = get_all_samples()
    total_count = len(samples)
    malicious_count = sum(1 for s in samples if s['is_malicious'] == 1)
    benign_count = total_count - malicious_count
    return {
        'total_samples': total_count,
        'malicious_samples': malicious_count,
        'benign_samples': benign_count,
        'detection_rate': round(malicious_count / total_count * 100, 2) if total_count else 0,
    }

@app.post("/api/upload")
async def upload_sample(file: UploadFile = File(...)):

    """上传并分析样本"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    try:
        # 异步读取，保存文件
        file_content = await file.read()

        print(f'file_name: {file.filename}')
         # 计算文件哈希
        file_hash = hashlib.md5(file_content).hexdigest() 
        
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file_hash)

        with open(file_path, "wb") as f:  # 使用同步写入
            f.write(file_content)

        file_size = len(file_content)
        
        # 模拟分类结果
        is_malicious = 1#random.choice([0, 1])
        classification_result = "恶意软件" if is_malicious else "正常软件"
        
        # 插入样本基本信息
        sample_id = insert_sample(
            file_hash=file_hash,
            file_name=file.filename,
            file_size=file_size,
            is_malicious=is_malicious,
            classification_result=classification_result
        )
        print(f'sample_id: {sample_id}')
        
        # 提取特征并插入数据库
        # 1. 直方图特征
        #byte_histogram = [random.randint(0, 10000) for _ in range(256)]
        #entropy_histogram = [random.randint(0, 10000) for _ in range(256)]
        byte_histogram = list(ByteHistogram().raw_features(file_content, None))
        entropy_histogram = list(ByteEntropyHistogram().raw_features(file_content, None))
        print(byte_histogram)
        print(entropy_histogram)
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

        lief_data = generate_mock_lief_data(is_malicious)
        dos_header = lief_data['dos_header']
        pe_header = lief_data['header']
        sections = lief_data['sections']
        imports = lief_data['imports']
        tls_info =lief_data.get('tls', {})
        resources = lief_data.get('resources', [])

        insert_lief_features(
            sample_id, dos_header, pe_header,
            sections, imports, tls_info, resources
        )
        
        return {
            'success': True,
            'sample_id': sample_id,
            'message': '已成功上传并分析样本'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

@app.get("/api/model/analysis")
async def get_model_analysis():
    """获取模型分析数据"""
    try:
        # 读取模型训练历史
        # with open("../machine_learning/models/histogram_history.pkl", "rb") as f:
        #     histogram_history = pickle.load(f)
        
        # 读取PE静态特征模型性能
        with open("../machine_learning/models/raw_feature_names.pkl", "rb") as f:
            model_names = pickle.load(f)
        
        # 读取特征工程模型性能
        # with open("../machine_learning/models/feature_importance.pkl", "rb") as f:
        #     feature_importance = pickle.load(f)
            
        # 生成模拟数据（实际项目中应该从文件读取）
        return {
            "histogram_model": {
                "metrics": {
                    "accuracy": 92.5,
                    "precision": 91.8,
                    "recall": 93.2,
                    "f1_score": 92.5,
                    "confusion_matrix": [[450, 35], [40, 475]]
                },
                # "training_history": histogram_history
            },
            "pe_raw_models": [
                {
                    "model_name": name,
                    "accuracy": random.uniform(85, 95),
                    "importance": [
                        {"feature": f"feature_{i}", "score": random.random()}
                        for i in range(10)
                    ]
                }
                for name in model_names
            ],
            "feature_engineering": {
                "metrics": {
                    "accuracy": 94.2,
                    "precision": 93.8,
                    "recall": 94.6,
                    "f1_score": 94.2,
                    "confusion_matrix": [[470, 25], [30, 475]]
                },
                # "feature_importance": feature_importance
            },
            "ensemble_metrics": {
                "accuracy": 96.98,
                "precision": 94.1,
                "recall": 95.5,
                "f1_score": 93.4,
                "confusion_matrix": [[11707, 52], [18, 5890]]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型分析数据失败: {str(e)}")
