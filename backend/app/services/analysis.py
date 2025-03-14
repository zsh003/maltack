import os
import hashlib
import magic
import yara
import pefile
import re
from datetime import datetime
import subprocess
import json
from app.config import Config

def get_basic_info(filepath):
    """获取文件基础信息"""
    with open(filepath, 'rb') as f:
        content = f.read()
        
    return {
        'filename': os.path.basename(filepath),
        'file_size': os.path.getsize(filepath),
        'file_type': magic.from_file(filepath),
        'mime_type': magic.from_file(filepath, mime=True),
        'md5': hashlib.md5(content).hexdigest(),
        'sha1': hashlib.sha1(content).hexdigest(),
        'sha256': hashlib.sha256(content).hexdigest(),
        'analyze_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def analyze_pe(filepath):
    """分析PE文件"""
    try:
        pe = pefile.PE(filepath)
        
        info = {
            'machine_type': hex(pe.FILE_HEADER.Machine),
            'timestamp': datetime.fromtimestamp(pe.FILE_HEADER.TimeDateStamp).strftime('%Y-%m-%d %H:%M:%S'),
            'subsystem': pe.OPTIONAL_HEADER.Subsystem,
            'dll_characteristics': pe.OPTIONAL_HEADER.DllCharacteristics,
            'sections': [],
            'imports': [],
            'exports': []
        }
        
        # 分析节区
        for section in pe.sections:
            info['sections'].append({
                'name': section.Name.decode().rstrip('\x00'),
                'virtual_address': hex(section.VirtualAddress),
                'virtual_size': hex(section.Misc_VirtualSize),
                'raw_size': hex(section.SizeOfRawData)
            })
        
        # 分析导入表
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_imports = {
                    'dll': entry.dll.decode(),
                    'functions': []
                }
                for imp in entry.imports:
                    if imp.name:
                        dll_imports['functions'].append(imp.name.decode())
                info['imports'].append(dll_imports)
        
        # 分析导出表
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name:
                    info['exports'].append({
                        'name': exp.name.decode(),
                        'address': hex(pe.OPTIONAL_HEADER.ImageBase + exp.address)
                    })
        
        return info
    except Exception as e:
        return {'error': str(e)}

def analyze_yara(filepath):
    """YARA规则匹配分析"""
    try:
        if not os.path.exists(Config.YARA_RULES_PATH):
            return {'error': 'YARA规则文件不存在'}
            
        rules = yara.compile(Config.YARA_RULES_PATH)
        matches = rules.match(filepath)
        
        return [{
            'rule_name': match.rule,
            'strings': [{
                'identifier': string.identifier,
                'data': string.data.hex(),
                'offset': string.offset
            } for string in match.strings],
            'tags': match.tags,
            'meta': match.meta
        } for match in matches]
    except Exception as e:
        return {'error': str(e)}

def analyze_sigma(filepath):
    """SIGMA规则匹配分析"""
    try:
        if not os.path.exists(Config.SIGMA_RULES_PATH):
            return {'error': 'SIGMA规则文件不存在'}
            
        # 使用sigma-cli工具进行规则匹配
        cmd = ['sigma-cli', 'scan', '-r', Config.SIGMA_RULES_PATH, filepath]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {'error': result.stderr}
    except Exception as e:
        return {'error': str(e)}

def analyze_strings(filepath):
    """字符串分析"""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            
        # 提取ASCII字符串
        ascii_pattern = rb'[\x20-\x7E]{4,}'
        ascii_strings = []
        for match in re.finditer(ascii_pattern, content):
            try:
                string = match.group().decode('ascii')
                ascii_strings.append({
                    'offset': match.start(),
                    'string': string
                })
            except UnicodeDecodeError:
                continue
                
        # 提取Unicode字符串
        unicode_pattern = rb'(?:[\x00-\x7F][\x00]){4,}'
        unicode_strings = []
        for match in re.finditer(unicode_pattern, content):
            try:
                string = match.group().decode('utf-16le')
                unicode_strings.append({
                    'offset': match.start(),
                    'string': string
                })
            except UnicodeDecodeError:
                continue
                
        return {
            'ascii_strings': ascii_strings,
            'unicode_strings': unicode_strings
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_file(filepath):
    """综合分析文件"""
    basic_info = get_basic_info(filepath)
    
    pe_info = {}
    if basic_info['file_type'].startswith('PE'):
        pe_info = analyze_pe(filepath)
    
    yara_matches = analyze_yara(filepath)
    
    sigma_matches = analyze_sigma(filepath)
    
    string_info = analyze_strings(filepath)
    
    result = {
        'basic_info': basic_info,
        'pe_info': pe_info,
        'yara_matches': yara_matches,
        'sigma_matches': sigma_matches,
        'string_info': string_info
    }

    return result



