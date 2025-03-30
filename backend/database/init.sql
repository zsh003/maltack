-- 字节直方图特征表
CREATE TABLE IF NOT EXISTS byte_histogram (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id VARCHAR(64) NOT NULL,
    byte_value INT NOT NULL,
    count INT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
);

-- 字节熵直方图特征表
CREATE TABLE IF NOT EXISTS byte_entropy (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id VARCHAR(64) NOT NULL,
    byte_value INT NOT NULL,
    entropy_value FLOAT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
);

-- PE静态特征表
CREATE TABLE IF NOT EXISTS pe_static_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id VARCHAR(64) NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    feature_data JSON NOT NULL,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
);

-- 特征工程表
CREATE TABLE IF NOT EXISTS feature_engineering (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id VARCHAR(64) NOT NULL,
    section_info JSON,
    string_matches JSON,
    yara_matches JSON,
    opcode_features JSON,
    boolean_features JSON,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
); 