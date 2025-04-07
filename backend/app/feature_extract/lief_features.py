import lief
import json
from typing import Dict, Any, List

class LiefFeatureExtractor:
    def extract_features(self, file_path: str) -> Dict[str, Any]:
        try:
            binary = lief.parse(file_path)
            if binary is None:
                return self._get_empty_features()
            
            return {
                "header_info": self._extract_header_info(binary),
                "sections": self._extract_sections(binary),
                "imports": self._extract_imports(binary),
                "exports": self._extract_exports(binary),
                "libraries": self._extract_libraries(binary),
                "tls": self._extract_tls(binary),
                "signatures": self._extract_signatures(binary)
            }
        except Exception as e:
            print(f"LIEF分析错误: {str(e)}")
            return self._get_empty_features()
    
    def _extract_header_info(self, binary) -> Dict[str, Any]:
        header = binary.header
        return {
            "machine": str(header.machine),
            "time_date_stamps": header.time_date_stamps,
            "sizeof_headers": header.sizeof_headers,
            "characteristics": [str(c) for c in header.characteristics_list],
            "magic": str(binary.optional_header.magic),
            "subsystem": str(binary.optional_header.subsystem),
            "dll_characteristics": [str(c) for c in binary.optional_header.dll_characteristics_lists]
        }
    
    def _extract_sections(self, binary) -> List[Dict[str, Any]]:
        return [{
            "name": section.name,
            "size": section.size,
            "virtual_size": section.virtual_size,
            "characteristics": [str(c) for c in section.characteristics_lists],
            "entropy": section.entropy,
            "is_executable": section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE),
            "is_writable": section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.MEM_WRITE),
            "is_readable": section.has_characteristic(lief.PE.SECTION_CHARACTERISTICS.MEM_READ)
        } for section in binary.sections]
    
    def _extract_imports(self, binary) -> List[Dict[str, Any]]:
        imports = []
        if binary.has_imports:
            for imp in binary.imports:
                entries = []
                for entry in imp.entries:
                    entries.append({
                        "name": entry.name if entry.name else "unknown",
                        "hint": entry.hint if hasattr(entry, "hint") else 0
                    })
                imports.append({
                    "library": imp.name,
                    "entries": entries
                })
        return imports
    
    def _extract_exports(self, binary) -> List[Dict[str, Any]]:
        exports = []
        if binary.has_exports:
            for exp in binary.exported_functions:
                exports.append({
                    "name": exp.name if exp.name else "unknown",
                    "address": exp.address,
                    "is_forwarded": exp.is_forwarded
                })
        return exports
    
    def _extract_libraries(self, binary) -> List[str]:
        return [lib.name for lib in binary.libraries] if binary.has_imports else []
    
    def _extract_tls(self, binary) -> Dict[str, Any]:
        if binary.has_tls:
            tls = binary.tls
            return {
                "callbacks": len(tls.callbacks),
                "address_of_raw_data": [tls.addressof_raw_data.start, tls.addressof_raw_data.end],
                "address_of_index": tls.addressof_index,
                "size_of_zero_fill": tls.sizeof_zero_fill,
                "characteristics": tls.characteristics
            }
        return {}
    
    def _extract_signatures(self, binary) -> Dict[str, Any]:
        if binary.has_signatures:
            return {
                "has_signature": True,
                "signature_info": {
                    "version": binary.signatures[0].version,
                    "digest_algorithm": str(binary.signatures[0].digest_algorithm),
                    "content_info": {
                        "content_type": str(binary.signatures[0].content_info.content_type),
                        "digest_algorithm": str(binary.signatures[0].content_info.digest_algorithm)
                    }
                }
            }
        return {"has_signature": False}
    
    def _get_empty_features(self) -> Dict[str, Any]:
        return {
            "header_info": {},
            "sections": [],
            "imports": [],
            "exports": [],
            "libraries": [],
            "tls": {},
            "signatures": {"has_signature": False}
        } 