import sqlite3
import os
import json
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'malware_features.db')

def init_db():
    """初始化数据库，创建必要的表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 样本基本信息表
    cursor.execute('DROP TABLE IF EXISTS samples')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_hash TEXT UNIQUE,
        file_name TEXT,
        file_size INTEGER,
        is_malicious INTEGER,
        analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        classification_result TEXT
    )
    ''')
    
    # 直方图特征表
    cursor.execute('DROP TABLE IF EXISTS histogram_features')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS histogram_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id INTEGER,
        byte_histogram TEXT,
        entropy_histogram TEXT,
        FOREIGN KEY (sample_id) REFERENCES samples (id)
    )
    ''')
    
    # PE静态特征表
    cursor.execute('DROP TABLE IF EXISTS pe_features')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pe_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id INTEGER,
        general_info TEXT,
        header_info TEXT,
        section_info TEXT,
        exports_info TEXT,
        FOREIGN KEY (sample_id) REFERENCES samples (id)
    )
    ''')
    
    # 特征工程表
    cursor.execute('DROP TABLE IF EXISTS engineered_features')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS engineered_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id INTEGER,
        section_features TEXT,
        string_match TEXT,
        yara_match TEXT,
        string_count TEXT,
        opcode_features TEXT,
        FOREIGN KEY (sample_id) REFERENCES samples (id)
    )
    ''')

    # LIEF分析表
    cursor.execute('DROP TABLE IF EXISTS lief_features')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lief_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id INTEGER,
        dos_header TEXT,
        pe_header TEXT,
        sections TEXT,
        imports TEXT,
        tls_info TEXT,
        resources TEXT,
        FOREIGN KEY (sample_id) REFERENCES samples (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def insert_sample(file_hash, file_name, file_size, is_malicious=None, classification_result=None):
    """插入样本基本信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT OR IGNORE INTO samples (file_hash, file_name, file_size, is_malicious, classification_result)
    VALUES (?, ?, ?, ?, ?)
    ''', (file_hash, file_name, file_size, is_malicious, classification_result))
    
    sample_id = cursor.lastrowid
    if sample_id == 0:  # 已存在，获取ID
        cursor.execute('SELECT id FROM samples WHERE file_hash = ?', (file_hash,))
        sample_id = cursor.fetchone()[0]
    
    conn.commit()
    conn.close()
    
    return sample_id

def insert_histogram_features(sample_id, byte_histogram, entropy_histogram):
    """插入直方图特征"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    byte_hist_json = json.dumps(byte_histogram)
    entropy_hist_json = json.dumps(entropy_histogram)
    
    cursor.execute('''
    INSERT INTO histogram_features (sample_id, byte_histogram, entropy_histogram)
    VALUES (?, ?, ?)
    ''', (sample_id, byte_hist_json, entropy_hist_json))
    
    conn.commit()
    conn.close()

def insert_pe_features(sample_id, general_info, header_info, section_info, exports_info):
    """插入PE静态特征"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO pe_features (sample_id, general_info, header_info, section_info, exports_info)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        sample_id,
        json.dumps(general_info),
        json.dumps(header_info),
        json.dumps(section_info),
        json.dumps(exports_info)
    ))
    
    conn.commit()
    conn.close()

def insert_engineered_features(sample_id, section_features, string_match, yara_match, string_count, opcode_features):
    """插入特征工程特征"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO engineered_features (sample_id, section_features, string_match, yara_match, string_count, opcode_features)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        sample_id,
        json.dumps(section_features),
        json.dumps(string_match),
        json.dumps(yara_match),
        json.dumps(string_count),
        json.dumps(opcode_features)
    ))
    
    conn.commit()
    conn.close()

def insert_lief_features(sample_id, dos_header, pe_header, sections, imports, tls_info, resources):
    """插入LIEF分析特征"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    conn.execute('''
    INSERT INTO lief_features (sample_id, dos_header, pe_header, sections, imports, tls_info, resources)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        sample_id,
        json.dumps(dos_header),
        json.dumps(pe_header),
        json.dumps(sections),
        json.dumps(imports),
        json.dumps(tls_info),
        json.dumps(resources)
    ))

    conn.commit()
    conn.close()

def get_all_samples():
    """获取所有样本的基本信息"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM samples ORDER BY analysis_time DESC')
    samples = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return samples

def get_sample_by_id(sample_id):
    """根据ID获取样本详细信息"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 获取基本信息
    cursor.execute('SELECT * FROM samples WHERE id = ?', (sample_id,))
    sample = dict(cursor.fetchone())
    
    # 获取直方图特征
    cursor.execute('SELECT * FROM histogram_features WHERE sample_id = ?', (sample_id,))
    row = cursor.fetchone()
    if row:
        histogram = dict(row)
        sample['histogram_features'] = {
            'byte_histogram': json.loads(histogram['byte_histogram']),
            'entropy_histogram': json.loads(histogram['entropy_histogram'])
        }
    
    # 获取PE静态特征
    cursor.execute('SELECT * FROM pe_features WHERE sample_id = ?', (sample_id,))
    row = cursor.fetchone()
    if row:
        pe_features = dict(row)
        sample['pe_features'] = {
            'general_info': json.loads(pe_features['general_info']),
            'header_info': json.loads(pe_features['header_info']),
            'section_info': json.loads(pe_features['section_info']),
            'exports_info': json.loads(pe_features['exports_info'])
        }
    
    # 获取特征工程特征
    cursor.execute('SELECT * FROM engineered_features WHERE sample_id = ?', (sample_id,))
    row = cursor.fetchone()
    if row:
        eng_features = dict(row)
        sample['engineered_features'] = {
            'section_features': json.loads(eng_features['section_features']),
            'string_match': json.loads(eng_features['string_match']),
            'yara_match': json.loads(eng_features['yara_match']),
            'string_count': json.loads(eng_features['string_count']),
            'opcode_features': json.loads(eng_features['opcode_features'])
        }
    
    # 获取LIEF特征
    cursor.execute('SELECT * FROM lief_features WHERE sample_id = ?', (sample_id,))
    row = cursor.fetchone()
    if row:
        lief_features = dict(row)
        sample['lief_features'] = {
            'dos_header': json.loads(lief_features['dos_header']),
            'header': json.loads(lief_features['pe_header']),
            'sections': json.loads(lief_features['sections']),
            'imports': json.loads(lief_features['imports']),
            'tls_info': json.loads(lief_features['tls_info']),
            'resources': json.loads(lief_features['resources'])
        }
    
    conn.close()
    return sample