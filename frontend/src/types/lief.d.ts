export interface DosHeader {
  magic: string;
  used_bytes_in_last_page: number;
  file_size_in_pages: number;
  number_of_relocation: number;
  header_size_in_paragraphs: number;
  min_extra_paragraphs: number;
  max_extra_paragraphs: number;
  initial_relative_ss: number;
  initial_sp: number;
  checksum: number;
  initial_ip: number;
  initial_relative_cs: number;
  address_of_relocation_table: number;
  overlay_number: number;
  oem_id: number;
  oem_info: number;
  address_of_new_exe_header: number;
}

export interface OptionalHeader {
  magic: string;
  major_linker_version: number;
  minor_linker_version: number;
  size_of_code: number;
  size_of_initialized_data: number;
  size_of_uninitialized_data: number;
  entry_point: number;
  base_of_code: number;
  base_of_data: number;
  image_base: number;
  section_alignment: number;
  file_alignment: number;
  major_operating_system_version: number;
  minor_operating_system_version: number;
  major_image_version: number;
  minor_image_version: number;
  major_subsystem_version: number;
  minor_subsystem_version: number;
  win32_version_value: number;
  size_of_image: number;
  size_of_headers: number;
  checksum: number;
  subsystem: string;
  dll_characteristics: string[];
  size_of_stack_reserve: number;
  size_of_stack_commit: number;
  size_of_heap_reserve: number;
  size_of_heap_commit: number;
  loader_flags: number;
  number_of_rva_and_size: number;
}

export interface Section {
  name: string;
  virtual_size: number;
  virtual_address: number;
  size_of_raw_data: number;
  pointer_to_raw_data: number;
  pointer_to_relocations: number;
  pointer_to_line_numbers: number;
  number_of_relocations: number;
  number_of_lines: number;
  characteristics: string[];
  entropy: number;
  is_readable: boolean;
  is_writable: boolean;
  is_executable: boolean;
}

export interface ImportEntry {
  name: string;
  hint: number;
}

export interface Import {
  library: string;
  entries: ImportEntry[];
}

export interface Export {
  name: string;
  address: string;
  is_forwarded: boolean;
}

export interface TLS {
  callbacks: number;
  address_of_raw_data: [number, number];
  address_of_index: number;
  size_of_zero_fill: number;
  characteristics: number;
}

export interface SignatureInfo {
  version: number;
  digest_algorithm: string;
  content_info: {
    content_type: string;
    digest_algorithm: string;
  };
}

export interface Resource {
  type: string;
  id: string;
  language: string;
  sublanguage: string;
  content?: string;
}

export interface LiefAnalysis {
  dos_header: DosHeader;
  header: {
    signature: string;
    machine: string;
    number_of_sections: number;
    time_date_stamps: number;
    pointer_to_symbol_table: number;
    number_of_symbols: number;
    size_of_optional_header: number;
    characteristics: string[];
  };
  optional_header: OptionalHeader;
  sections: Section[];
  imports: Import[];
  exports: Export[];
  tls?: TLS;
  signatures: {
    has_signature: boolean;
    signature_info?: SignatureInfo;
  };
  resources?: Resource[];
  manifest?: string;
} 