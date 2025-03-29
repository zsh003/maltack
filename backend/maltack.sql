USE maltack;

DROP TABLE IF EXISTS upload_history;
CREATE TABLE upload_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    environment VARCHAR(255),
    upload_time DATETIME NOT NULL,
    threat_level VARCHAR(20),
    status VARCHAR(20) DEFAULT 'pending' -- 状态：pending, analyzing, completed
);