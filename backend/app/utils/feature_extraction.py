import lief
import numpy as np
from collections import Counter
import re
import yara
import os

from .raw_features import ByteHistogram, ByteEntropyHistogram, PEFeatureExtractor

def extract_byte_histogram(file_content):
    """提取字节直方图特征"""
    Histogram = ByteHistogram().raw_features(file_content, None)
    return {i: value for i, value in enumerate(list(Histogram))}

def extract_byte_entropy(file_content):
    """提取字节熵特征"""
    Byte_Entropy = ByteEntropyHistogram().raw_features(file_content, None)
    return {i: value for i, value in enumerate(list(Byte_Entropy))}

def extract_pe_static_features(filepath):
    """提取PE静态特征"""
    try:
        binary = lief.parse(filepath)
        if not binary:
            print(f"无法解析文件: {filepath}")
            return {}
        
        features = {}
        
        # 基本信息
        features['general_info'] = {
            'file_size': binary.header.sizeof_headers,
            'entry_point': binary.entrypoint,
            'machine_type': str(binary.header.machine),
            'timestamp': binary.header.time_date_stamps,
        }
        
        # 节区信息
        features['section_info'] = {
            'sections': [{
                'name': section.name,
                'size': section.size,
                'entropy': section.entropy,
                'vsize': section.virtual_size,
                'props': [str(prop) for prop in section.characteristics_list]
            } for section in binary.sections]
        }
        
        # 导出函数信息
        if binary.has_exports:
            features['export_info'] = {
                'exports': [{
                    'name': exp.name,
                    'address': exp.address,
                    'ordinal': exp.ordinal
                } for exp in binary.exported_functions]
            }
        
        return features
    except Exception as e:
        print(f"提取PE静态特征失败: {str(e)}")
        return {}

def extract_feature_engineering(filepath):
    """提取特征工程数据"""
    try:
        binary = lief.parse(filepath)
        if not binary:
            return {}, [], [], [], {}
        
        # 1. 节区信息特征
        section_info = {}
        for section in binary.sections:
            section_info[f'section_{section.name}_size'] = section.size
            section_info[f'section_{section.name}_entropy'] = section.entropy
            section_info[f'section_{section.name}_vsize'] = section.virtual_size
        
        # 2. 字符串模式匹配
        string_matches = []
        with open(filepath, 'rb') as f:
            content = f.read()
            
            # 匹配URL
            urls = re.findall(b'https?://[^\s<>"]+|www\.[^\s<>"]+', content)
            if urls:
                string_matches.append({
                    'type': 'url',
                    'count': len(urls),
                    'examples': [url.decode() for url in urls[:3]]
                })
            
            # 匹配IP地址
            ips = re.findall(b'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', content)
            if ips:
                string_matches.append({
                    'type': 'ip',
                    'count': len(ips),
                    'examples': [ip.decode() for ip in ips[:3]]
                })
        
        # 3. YARA规则匹配
        yara_matches = []
        rules_path = os.path.join(os.path.dirname(__file__), '../../rules')
        rules = yara.compile(rules_path)
        matches = rules.match(filepath)
        for match in matches:
            yara_matches.append({
                'name': match.rule,
                'matched': True,
                'description': match.meta.get('description', '')
            })
        
        # 4. 操作码特征
        opcode_features = []
        if binary.has_imports:
            for imp in binary.imports:
                for entry in imp.entries:
                    opcode_features.append({
                        'opcode': entry.name,
                        'count': 1,
                        'frequency': 1.0 / len(binary.imports)
                    })
        
        # 5. 布尔特征
        boolean_features = {
            'has_imports': binary.has_imports,
            'has_exports': binary.has_exports,
            'has_debug': binary.has_debug,
            'has_tls': binary.has_tls,
            'has_resources': binary.has_resources,
            'has_signature': binary.has_signature,
            'has_configuration': binary.has_configuration,
        }
        
        return section_info, string_matches, yara_matches, opcode_features, boolean_features
    except Exception as e:
        print(f"提取特征工程数据失败: {str(e)}")
        return {}, [], [], [], {} 