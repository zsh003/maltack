from sqlalchemy import Column, Integer, String, Float, DateTime, BIGINT
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Float
from sqlalchemy.orm import relationship, sessionmaker

from app import db


class User(db.Model):
    __tablename__ = 'user'
    userid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.password}')"


class UploadHistory(db.Model):
    __tablename__ = 'upload_history'
    file_id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True)
    file_url = db.Column(db.String(200), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    environment = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, nullable=False)
    threat_level = db.Column(db.String(20), default='unknown', nullable=False)
    status = db.Column(db.String(20), default='pending', nullable=False)

    def __repr__(self):
        return f'<UploadHistory {self.file_name}>'

class FileInfo(db.Model):
    __tablename__ = 'basic_info'
    file_id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False)
    file_size = Column(Float, nullable=False)
    file_type = Column(String(255))
    mime_type = Column(String(100))
    analyze_time = Column(DateTime)
    md5 = Column(String(32))
    sha1 = Column(String(40))
    sha256 = Column(String(64))
    upload_history = relationship("UploadHistory", back_populates="basic_infos")

UploadHistory.basic_info = relationship("FileInfo", uselist=False, back_populates="upload_history")

class PEInfo(db.Model):
    __tablename__ = 'pe_info'
    file_id = Column(Integer, ForeignKey('upload_history.file_id'), primary_key=True)
    machine_type = Column(String(10))
    timestamp = Column(DateTime)
    subsystem = Column(String(50))
    dll_characteristics = Column(String(50))
    sections = Column(Text) # JSON string or similar for storing complex structures
    imports = Column(Text)
    exports = Column(Text)
    upload_history = relationship("UploadHistory", back_populates="pe_infos")

UploadHistory.pe_infos = relationship("PEInfo", uselist=False, back_populates="upload_history")

class YaraMatch(db.Model):
    __tablename__ = 'yara_match'
    file_id = Column(Integer, ForeignKey('upload_history.file_id'), primary_key=True)
    rule_name = Column(String(255))
    strings = Column(Text)
    tags = Column(Text)
    meta = Column(Text)
    upload_history = relationship("UploadHistory", back_populates="yara_matches")

UploadHistory.yara_matches = relationship("YaraMatch", uselist=False, back_populates="upload_history")

class SigmaMatch(db.Model):
    __tablename__ = 'sigma_match'
    file_id = Column(Integer, ForeignKey('upload_history.file_id'), primary_key=True)
    rule_details = Column(Text)
    upload_history = relationship("UploadHistory", back_populates="sigma_matches")

UploadHistory.sigma_matches = relationship("SigmaMatch", uselist=False, back_populates="upload_history")

class AnalyzeStrings(db.Model):
    __tablename__ = 'analyze_strings'
    file_id = Column(Integer, ForeignKey('upload_history.file_id'), primary_key=True)
    ascii_strings = Column(Text)
    unicode_strings = Column(Text)
    upload_history = relationship("UploadHistory", back_populates="analyze_strings")

UploadHistory.analyze_strings = relationship("AnalyzeStrings", uselist=False, back_populates="upload_history")
