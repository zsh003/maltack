import os
import random
import hashlib
from app.database import (
    init_db, insert_sample, insert_histogram_features, 
    insert_pe_features, insert_engineered_features
)

# 初始化数据库
init_db()

# 生成模拟数据的函数
def generate_mock_sample(index):
    # 生成模拟文件信息
    file_name = f"sample_{index}.exe"
    file_hash = hashlib.md5(f"mock_content_{index}".encode()).hexdigest()
    file_size = random.randint(10000, 10000000)
    is_malicious = random.choice([0, 1])
    classification_result = "恶意软件" if is_malicious else "正常软件"
    
    # 插入样本基本信息
    sample_id = insert_sample(
        file_hash=file_hash,
        file_name=file_name,
        file_size=file_size,
        is_malicious=is_malicious,
        classification_result=classification_result
    )
    
    # 生成直方图特征
    byte_histogram = [random.randint(0, 10000) for _ in range(256)]
    entropy_histogram = [random.randint(0, 10000) for _ in range(256)]
    insert_histogram_features(sample_id, byte_histogram, entropy_histogram)
    
    # 生成PE静态特征
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
        "machine_type": "I386" if random.random() < 0.7 else "AMD64",
        "timestamp": "2022-01-01",
        "num_sections": random.randint(3, 10),
        "pointer_symbol_table": random.randint(0, 1000000),
        "characteristics": ["EXECUTABLE", "32BIT"] if random.random() < 0.7 else ["EXECUTABLE", "DLL", "64BIT"]
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
            },
            {
                "name": ".rsrc",
                "size": random.randint(1000, 50000),
                "entropy": round(random.uniform(2.0, 7.0), 2),
                "vsize": random.randint(1000, 50000),
                "props": ["CNT_INITIALIZED_DATA", "MEM_READ"]
            }
        ]
    }
    
    # 恶意软件可能会有额外的可疑部分
    if is_malicious:
        section_info["sections"].append({
            "name": random.choice([".evil", ".xyz", ".crypt", ".packed"]),
            "size": random.randint(5000, 200000),
            "entropy": round(random.uniform(7.0, 8.0), 2),
            "vsize": random.randint(5000, 200000),
            "props": ["MEM_EXECUTE", "MEM_READ", "MEM_WRITE"]
        })
    
    exports_info = {
        "exports": [
            {"name": f"Function{i}", "address": f"0x{random.randint(4096, 8192000):x}"}
            for i in range(random.randint(0, 10))
        ]
    }
    
    insert_pe_features(sample_id, general_info, header_info, section_info, exports_info)
    
    # 生成特征工程特征
    # 计算节区大小总和用于权重计算
    total_size = random.randint(50000, 1500000)
    size_R = random.randint(10000, 500000)
    size_W = random.randint(10000, 500000)
    size_X = random.randint(10000, 500000)
    
    # 计算权重
    size_R_weight = round(size_R / total_size, 4)
    size_W_weight = round(size_W / total_size, 4)
    size_X_weight = round(size_X / total_size, 4)
    
    # 生成熵值
    entr_R = round(random.uniform(5.0, 7.0), 2)
    entr_W = round(random.uniform(2.0, 5.0), 2)
    entr_X = round(random.uniform(6.0, 8.0), 2)
    
    # 计算熵值权重
    total_entropy = entr_R + entr_W + entr_X
    entr_R_weight = round(entr_R / total_entropy, 8)
    entr_W_weight = round(entr_W / total_entropy, 8)
    entr_X_weight = round(entr_X / total_entropy, 8)
    
    section_features = {
        "entry": random.randint(0, 10),
        "size_R": size_R,
        "size_W": size_W,
        "size_X": size_X,
        "entr_R": entr_R,
        "entr_W": entr_W,
        "entr_X": entr_X,
        "size_R_weight": size_R_weight,
        "size_W_weight": size_W_weight,
        "size_X_weight": size_X_weight,
        "entr_R_weight": entr_R_weight,
        "entr_W_weight": entr_W_weight,
        "entr_X_weight": entr_X_weight,
        "rsrc_num": random.randint(0, 3),
        "section_num": random.randint(3, 10),
        "file_size": file_size
    }
    
    # 恶意软件更有可能包含可疑字符串
    string_match = {
        "mz_count": random.randint(5, 15),
        "mz_mean": round(random.uniform(2, 3), 2),
        "pe_count": random.randint(10, 20),
        "pe_mean": round(random.uniform(2, 3), 2),
        "pool_count": random.randint(5, 10) if is_malicious else random.randint(0, 3),
        "pool_mean": round(random.uniform(3, 5), 2),
        "cpu_count": random.randint(5, 15) if is_malicious else random.randint(0, 5),
        "cpu_mean": round(random.uniform(2, 4), 2),
        "gpu_count": random.randint(0, 5) if is_malicious else 0,
        "gpu_mean": round(random.uniform(2, 4), 2) if is_malicious else 0,
        "coin_count": random.randint(0, 10) if is_malicious else 0,
        "coin_mean": round(random.uniform(2, 4), 2) if is_malicious else 0,
        "btc_count": random.randint(0, 5) if is_malicious else random.randint(0, 2),
        "btc_mean": round(random.uniform(20, 30), 2),
        "ltc_count": random.randint(0, 5) if is_malicious else 0,
        "ltc_mean": round(random.uniform(20, 30), 2),
        "xmr_count": random.randint(0, 5) if is_malicious else 0,
        "xmr_mean": round(random.uniform(20, 30), 2),
        "paths_count": random.randint(3, 20),
        "paths_mean": round(random.uniform(3, 5), 2),
        "regs_count": random.randint(0, 20) if is_malicious else random.randint(0, 5),
        "regs_mean": round(random.uniform(2, 4), 2),
        "urls_count": random.randint(2, 20) if is_malicious else random.randint(0, 5),
        "urls_mean": round(random.uniform(3, 6), 2),
        "ips_count": random.randint(0, 10) if is_malicious else random.randint(0, 3),
        "ips_mean": round(random.uniform(6, 8), 2)
    }
    
    yara_match = {
        "packer_count": random.randint(1, 3) if is_malicious else 0,
        "yargen_count": random.randint(1, 5) if is_malicious else random.randint(0, 2)
    }
    
    string_count = {
        "av_count": random.randint(0, 10) if is_malicious else 0,
        "dbg_count": random.randint(0, 5),
        "pool_name_count": random.randint(0, 20),
        "algorithm_name_count": random.randint(0, 5),
        "coin_name_count": random.randint(0, 10) if is_malicious else 0
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
    
    return sample_id

# 生成30个模拟样本
def create_mock_data(count=30):
    print(f"开始生成{count}个模拟样本数据...")
    for i in range(1, count+1):
        sample_id = generate_mock_sample(i)
        print(f"生成样本 {i}/{count}, ID: {sample_id}")
    print("模拟数据生成完成！")

if __name__ == "__main__":
    create_mock_data(30) 