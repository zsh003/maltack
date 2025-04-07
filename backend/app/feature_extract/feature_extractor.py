import pefile
import numpy as np
import hashlib
import re
import os
from typing import Dict, Any

def calculate_entropy(data: bytes) -> float:
    """计算数据的熵值"""
    if not data:
        return 0.0
    
    # 计算字节频率
    counts = np.zeros(256, dtype=np.uint32)
    for byte in data:
        counts[byte] += 1
    
    # 计算概率
    probs = counts[counts > 0] / len(data)
    
    # 计算熵值
    return -np.sum(probs * np.log2(probs))

def extract_histogram_features(pe: pefile.PE) -> Dict[str, Any]:
    """提取直方图特征"""
    # 获取所有节区的数据
    all_data = b''
    for section in pe.sections:
        all_data += section.get_data()
    
    # 计算字节直方图
    byte_histogram = np.zeros(256, dtype=np.uint32)
    for byte in all_data:
        byte_histogram[byte] += 1
    
    # 计算熵值直方图
    entropy_histogram = np.zeros(256, dtype=np.float32)
    for i in range(0, len(all_data), 256):
        block = all_data[i:i+256]
        entropy = calculate_entropy(block)
        entropy_histogram[i//256] = entropy
    
    return {
        'byte_histogram': byte_histogram.tolist(),
        'entropy_histogram': entropy_histogram.tolist()
    }

def extract_pe_features(pe: pefile.PE) -> Dict[str, Any]:
    """提取PE静态特征"""
    # 通用信息
    general_info = {
        'debug_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].Size,
        'tls_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[9].Size,
        'relocations_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[5].Size,
        'major_version': pe.OPTIONAL_HEADER.MajorLinkerVersion,
        'minor_version': pe.OPTIONAL_HEADER.MinorLinkerVersion,
        'num_data_directories': len(pe.OPTIONAL_HEADER.DATA_DIRECTORY)
    }
    
    # 头部信息
    header_info = {
        'pe_signature': hex(pe.NT_HEADERS.Signature),
        'machine_type': hex(pe.FILE_HEADER.Machine),
        'timestamp': pe.FILE_HEADER.TimeDateStamp,
        'num_sections': pe.FILE_HEADER.NumberOfSections,
        'pointer_symbol_table': pe.FILE_HEADER.PointerToSymbolTable,
        'characteristics': [hex(flag) for flag in pe.FILE_HEADER.Characteristics]
    }
    
    # 节区信息
    sections = []
    for section in pe.sections:
        section_data = section.get_data()
        sections.append({
            'name': section.Name.decode().rstrip('\x00'),
            'size': section.SizeOfRawData,
            'entropy': calculate_entropy(section_data),
            'vsize': section.Misc_VirtualSize,
            'props': [
                'CNT_CODE' if section.Characteristics & 0x20 else '',
                'MEM_EXECUTE' if section.Characteristics & 0x20000000 else '',
                'MEM_READ' if section.Characteristics & 0x40000000 else '',
                'MEM_WRITE' if section.Characteristics & 0x80000000 else ''
            ]
        })
    
    section_info = {
        'entry': sections[0]['name'] if sections else '',
        'sections': sections
    }
    
    # 导出函数信息
    exports = []
    if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
        for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            if exp.name:
                exports.append({
                    'name': exp.name.decode(),
                    'address': hex(exp.address)
                })
    
    exports_info = {
        'exports': exports
    }
    
    return {
        'general_info': general_info,
        'header_info': header_info,
        'section_info': section_info,
        'exports_info': exports_info
    }

def extract_engineered_features(pe: pefile.PE) -> Dict[str, Any]:
    """提取特征工程特征"""
    # 节区特征
    section_features = {
        'entry': 0,
        'size_R': 0,
        'size_W': 0,
        'size_X': 0,
        'entr_R': 0.0,
        'entr_W': 0.0,
        'entr_X': 0.0,
        'rsrc_num': 0,
        'section_num': len(pe.sections),
        'file_size': os.path.getsize(pe.filename)
    }
    
    # 计算节区特征
    total_size = 0
    for i, section in enumerate(pe.sections):
        section_data = section.get_data()
        size = section.SizeOfRawData
        entropy = calculate_entropy(section_data)
        
        if i == 0:
            section_features['entry'] = i
        
        if section.Characteristics & 0x40000000:  # MEM_READ
            section_features['size_R'] += size
            section_features['entr_R'] += entropy
        if section.Characteristics & 0x80000000:  # MEM_WRITE
            section_features['size_W'] += size
            section_features['entr_W'] += entropy
        if section.Characteristics & 0x20000000:  # MEM_EXECUTE
            section_features['size_X'] += size
            section_features['entr_X'] += entropy
        
        if '.rsrc' in section.Name.decode():
            section_features['rsrc_num'] += 1
        
        total_size += size
    
    # 计算权重
    if total_size > 0:
        section_features['size_R_weight'] = round(section_features['size_R'] / total_size, 4)
        section_features['size_W_weight'] = round(section_features['size_W'] / total_size, 4)
        section_features['size_X_weight'] = round(section_features['size_X'] / total_size, 4)
    
    # 计算熵值权重
    total_entropy = section_features['entr_R'] + section_features['entr_W'] + section_features['entr_X']
    if total_entropy > 0:
        section_features['entr_R_weight'] = round(section_features['entr_R'] / total_entropy, 8)
        section_features['entr_W_weight'] = round(section_features['entr_W'] / total_entropy, 8)
        section_features['entr_X_weight'] = round(section_features['entr_X'] / total_entropy, 8)
    
    # 字符串匹配特征
    string_match = {
        'mz_count': 0,
        'mz_mean': 0.0,
        'pe_count': 0,
        'pe_mean': 0.0,
        'pool_count': 0,
        'pool_mean': 0.0,
        'cpu_count': 0,
        'cpu_mean': 0.0,
        'gpu_count': 0,
        'gpu_mean': 0.0,
        'coin_count': 0,
        'coin_mean': 0.0
    }
    
    # 提取所有字符串
    all_strings = []
    for section in pe.sections:
        try:
            data = section.get_data()
            strings = re.findall(b'[\\x20-\\x7E]{4,}', data)
            all_strings.extend([s.decode() for s in strings])
        except:
            continue
    
    # 分析字符串
    for string in all_strings:
        if 'MZ' in string:
            string_match['mz_count'] += 1
            string_match['mz_mean'] += len(string)
        if 'PE' in string:
            string_match['pe_count'] += 1
            string_match['pe_mean'] += len(string)
        if 'pool' in string.lower():
            string_match['pool_count'] += 1
            string_match['pool_mean'] += len(string)
        if 'cpu' in string.lower():
            string_match['cpu_count'] += 1
            string_match['cpu_mean'] += len(string)
        if 'gpu' in string.lower():
            string_match['gpu_count'] += 1
            string_match['gpu_mean'] += len(string)
        if any(coin in string.lower() for coin in ['btc', 'eth', 'xmr']):
            string_match['coin_count'] += 1
            string_match['coin_mean'] += len(string)
    
    # 计算平均值
    for key in ['mz', 'pe', 'pool', 'cpu', 'gpu', 'coin']:
        count = string_match[f'{key}_count']
        if count > 0:
            string_match[f'{key}_mean'] = round(string_match[f'{key}_mean'] / count, 2)
    
    # YARA匹配特征
    yara_match = {
        'packer_count': 0,
        'yargen_count': 0
    }
    
    # 字符串计数特征
    string_count = {
        'av_count': 0,
        'dbg_count': 0,
        'pool_name_count': 0,
        'algorithm_name_count': 0,
        'coin_name_count': 0
    }
    
    # 操作码特征
    opcode_features = {
        'opcode_min': 0,
        'opcode_max': 0,
        'opcode_sum': 0,
        'opcode_mean': 0.0,
        'opcode_var': 0.0,
        'opcode_count': 0,
        'opcode_uniq': 0
    }
    
    return {
        'section_features': section_features,
        'string_match': string_match,
        'yara_match': yara_match,
        'string_count': string_count,
        'opcode_features': opcode_features
    }

def extract_features(file_path: str) -> Dict[str, Any]:
    """提取文件的所有特征"""
    try:
        pe = pefile.PE(file_path)
        
        # 提取各类特征
        histogram_features = extract_histogram_features(pe)
        pe_features = extract_pe_features(pe)
        engineered_features = extract_engineered_features(pe)
        
        # 判断是否为恶意软件（示例：基于特征组合的简单判断）
        is_malicious = (
            engineered_features['section_features']['size_X'] > 1000000 or
            engineered_features['string_match']['coin_count'] > 5 or
            engineered_features['string_match']['pool_count'] > 3
        )
        
        classification_result = "恶意软件" if is_malicious else "正常软件"
        
        return {
            'is_malicious': 1 if is_malicious else 0,
            'classification_result': classification_result,
            'histogram_features': histogram_features,
            'pe_features': pe_features,
            'engineered_features': engineered_features
        }
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return {
            'is_malicious': 0,
            'classification_result': "分析失败",
            'histogram_features': {'byte_histogram': [], 'entropy_histogram': []},
            'pe_features': {
                'general_info': {},
                'header_info': {},
                'section_info': {'entry': '', 'sections': []},
                'exports_info': {'exports': []}
            },
            'engineered_features': {
                'section_features': {},
                'string_match': {},
                'yara_match': {},
                'string_count': {},
                'opcode_features': {}
            }
        } 