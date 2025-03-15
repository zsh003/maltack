interface AnalysisResult {
  basic_info: {
    filename: string
    file_size: number
    file_type: string
    mime_type: string
    md5: string
    sha1: string
    sha256: string
    analyze_time: string
  }
  pe_info?: {
    machine_type: string
    timestamp: string
    subsystem: number
    dll_characteristics: number
    sections: Array<{
      name: string
      virtual_address: string
      virtual_size: string
      raw_size: string
    }>
    imports: Array<{
      dll: string
      functions: string[]
    }>
    exports: Array<{
      name: string
      address: string
    }>
  }
  yara_matches: Array<{
    rule_name: string
    strings: Array<{
      identifier: string
      data: string
      offset: number
    }>
    tags: string[]
    meta: Record<string, unknown>
  }>
  sigma_matches: Array<{
    id: string
    title: string
    description: string
    level: string
    tags: string[]
  }>
  string_info: {
    ascii_strings: Array<{
      offset: number
      string: string
    }>
    unicode_strings: Array<{
      offset: number
      string: string
    }>
  }
}
