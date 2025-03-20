-- 测试数据：上传历史记录
INSERT INTO upload_history (file_name, file_type, environment, upload_time, threat_level, status)
VALUES
('test_file_1.exe', 'EXEx64', 'Windows 10 (22H2 64bit)', '2025-03-20 10:00:00', 'low', 'completed'),
('test_file_2.exe', 'EXEx86', 'Windows 7 (SP1 2019全补丁版本 32bit)', '2025-03-21 12:30:00', 'medium', 'completed'),
('test_file_3', 'ELFx64', 'Linux (Ubuntu 17.04 64bit)', '2025-03-22 14:45:00', 'high', 'analyzing'),
('test_file_4.elf', 'ELFx64', 'Linux (CentOS 7.5 64bit)', '2025-03-25 14:10:00', 'high', 'analyzing'),
('test_file_5.dll', 'DLLx86', 'Windows 10 (1903 64bit)', '2025-03-27 19:45:00', 'medium', 'pending');
