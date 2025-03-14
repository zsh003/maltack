rule Suspicious_Strings {
    strings:
        $cmd1 = "cmd.exe" nocase
        $cmd2 = "powershell.exe" nocase
        $cmd3 = "wscript.shell" nocase
        $cmd4 = "shell.application" nocase
        $cmd5 = "regsvr32.exe" nocase
        $cmd6 = "certutil.exe" nocase
        $cmd7 = "bitsadmin.exe" nocase
        $cmd8 = "mshta.exe" nocase
        $cmd9 = "rundll32.exe" nocase
        $cmd10 = "reg.exe" nocase
        
    condition:
        any of them
}

rule Suspicious_Imports {
    strings:
        $import1 = "CreateRemoteThread" nocase
        $import2 = "VirtualAlloc" nocase
        $import3 = "WriteProcessMemory" nocase
        $import4 = "ShellExecute" nocase
        $import5 = "WinExec" nocase
        $import6 = "CreateProcess" nocase
        $import7 = "InternetOpen" nocase
        $import8 = "URLDownloadToFile" nocase
        $import9 = "RegCreateKey" nocase
        $import10 = "RegSetValue" nocase
        
    condition:
        any of them
}

rule Suspicious_Sections {
    strings:
        $section1 = ".text" nocase
        $section2 = ".data" nocase
        $section3 = ".rdata" nocase
        $section4 = ".bss" nocase
        $section5 = ".rsrc" nocase
        
    condition:
        any of them
}

rule Suspicious_Resources {
    strings:
        $resource1 = "RT_RCDATA" nocase
        $resource2 = "RT_BITMAP" nocase
        $resource3 = "RT_ICON" nocase
        $resource4 = "RT_VERSION" nocase
        $resource5 = "RT_MANIFEST" nocase
        
    condition:
        any of them
} 