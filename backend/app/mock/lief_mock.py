import random
import time
from typing import Dict, Any, List

def generate_mock_lief_data(is_malicious: bool = False) -> Dict[str, Any]:
    """生成模拟的LIEF分析数据"""
    current_time = int(time.time())
    
    # 生成基本节区
    sections = [
        {
            "name": ".text",
            "virtual_size": random.randint(1000, 100000),
            "virtual_address": 0x1000,
            "size_of_raw_data": random.randint(1000, 100000),
            "pointer_to_raw_data": 0x400,
            "pointer_to_relocations": 0,
            "pointer_to_line_numbers": 0,
            "number_of_relocations": 0,
            "number_of_lines": 0,
            "characteristics": ["CNT_CODE", "MEM_EXECUTE", "MEM_READ"],
            "entropy": round(random.uniform(5.0, 8.0), 2),
            "is_readable": True,
            "is_writable": False,
            "is_executable": True
        },
        {
            "name": ".data",
            "virtual_size": random.randint(1000, 50000),
            "virtual_address": 0x4000,
            "size_of_raw_data": random.randint(1000, 50000),
            "pointer_to_raw_data": 0x2600,
            "pointer_to_relocations": 0,
            "pointer_to_line_numbers": 0,
            "number_of_relocations": 0,
            "number_of_lines": 0,
            "characteristics": ["CNT_INITIALIZED_DATA", "MEM_READ", "MEM_WRITE"],
            "entropy": round(random.uniform(2.0, 6.0), 2),
            "is_readable": True,
            "is_writable": True,
            "is_executable": False
        }
    ]

    # 可疑的导入函数
    suspicious_imports = [
        ("kernel32.dll", ["VirtualAlloc", "VirtualProtect", "WriteProcessMemory"]),
        ("user32.dll", ["FindWindowA", "SetWindowsHookEx"]),
        ("advapi32.dll", ["CryptEncrypt", "CryptDecrypt"]),
        ("wininet.dll", ["InternetOpenA", "InternetConnectA"]),
    ]

    # 正常的导入函数
    normal_imports = [
        ("kernel32.dll", ["GetModuleHandle", "LoadLibrary", "GetProcAddress"]),
        ("user32.dll", ["MessageBox", "DialogBox"]),
        ("gdi32.dll", ["CreateFont", "SetBkColor"]),
        ("shell32.dll", ["ShellExecute", "SHGetFolderPath"]),
    ]

    # 根据是否恶意选择导入函数
    selected_imports = suspicious_imports if is_malicious else normal_imports
    imports = []
    for lib, funcs in selected_imports:
        entries = []
        for i, func in enumerate(funcs):
            entries.append({
                "name": func,
                "hint": i + 1
            })
        imports.append({
            "library": lib,
            "entries": entries
        })

    # 生成TLS数据（如果是恶意样本，增加回调函数数量）
    tls = {
        "callbacks": random.randint(3, 10) if is_malicious else random.randint(0, 2),
        "address_of_raw_data": [0x40b041, 0x40b044],
        "address_of_index": 0x40805c,
        "size_of_zero_fill": random.randint(0, 1000),
        "characteristics": random.randint(0, 0xFFFF)
    } if random.random() < 0.7 else None

    # 生成资源信息
    resources = [
        {
            "type": "VERSION",
            "id": "1",
            "language": "NEUTRAL",
            "sublanguage": "DEFAULT"
        },
        {
            "type": "MANIFEST",
            "id": "1",
            "language": "NEUTRAL",
            "sublanguage": "DEFAULT",
            "content": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n</assembly>'
        }
    ] if random.random() < 0.8 else None

    return {
        "dos_header": {
            "magic": "0x5A4D",
            "used_bytes_in_last_page": 0x90,
            "file_size_in_pages": random.randint(3, 10),
            "number_of_relocation": 0,
            "header_size_in_paragraphs": 4,
            "min_extra_paragraphs": 0,
            "max_extra_paragraphs": 0xFFFF,
            "initial_relative_ss": 0,
            "initial_sp": 0xB8,
            "checksum": 0,
            "initial_ip": 0,
            "initial_relative_cs": 0,
            "address_of_relocation_table": 0x40,
            "overlay_number": 0,
            "oem_id": 0,
            "oem_info": 0,
            "address_of_new_exe_header": 0x80
        },
        "header": {
            "signature": "50 45 00 00",
            "machine": "AMD64" if random.random() < 0.7 else "I386",
            "number_of_sections": len(sections),
            "time_date_stamps": current_time - random.randint(0, 365*24*3600),
            "pointer_to_symbol_table": 0,
            "number_of_symbols": 0,
            "size_of_optional_header": 0xF0,
            "characteristics": [
                "EXECUTABLE_IMAGE",
                "LARGE_ADDRESS_AWARE",
                *(['DYNAMIC_BASE', 'NX_COMPAT'] if not is_malicious else [])
            ]
        },
        "optional_header": {
            "magic": "0x20B",
            "major_linker_version": random.randint(2, 14),
            "minor_linker_version": random.randint(0, 30),
            "size_of_code": random.randint(1000, 100000),
            "size_of_initialized_data": random.randint(1000, 50000),
            "size_of_uninitialized_data": random.randint(0, 1000),
            "entry_point": 0x1000 + random.randint(0, 0x1000),
            "base_of_code": 0x1000,
            "base_of_data": 0,
            "image_base": 0x400000,
            "section_alignment": 0x1000,
            "file_alignment": 0x200,
            "major_operating_system_version": 6,
            "minor_operating_system_version": 1,
            "major_image_version": 0,
            "minor_image_version": 0,
            "major_subsystem_version": 6,
            "minor_subsystem_version": 1,
            "win32_version_value": 0,
            "size_of_image": 0x1000 * (len(sections) + 2),
            "size_of_headers": 0x400,
            "checksum": random.randint(0, 0xFFFFFFFF),
            "subsystem": "WINDOWS_GUI",
            "dll_characteristics": [
                "DYNAMIC_BASE",
                "NX_COMPAT",
                "TERMINAL_SERVER_AWARE"
            ] if not is_malicious else [],
            "size_of_stack_reserve": 0x100000,
            "size_of_stack_commit": 0x1000,
            "size_of_heap_reserve": 0x100000,
            "size_of_heap_commit": 0x1000,
            "loader_flags": 0,
            "number_of_rva_and_size": 16
        },
        "sections": sections,
        "imports": imports,
        "exports": [],  # 大多数可执行文件没有导出表
        "tls": tls,
        "signatures": {
            "has_signature": not is_malicious and random.random() < 0.7,
            "signature_info": {
                "version": 1,
                "digest_algorithm": "SHA256",
                "content_info": {
                    "content_type": "SPC_INDIRECT_DATA_CONTENT",
                    "digest_algorithm": "SHA256"
                }
            } if not is_malicious and random.random() < 0.7 else None
        },
        "resources": resources,
        "manifest": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n</assembly>' if random.random() < 0.8 else None
    } 