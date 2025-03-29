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
    file_url = db.Column(db.String(200))
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    environment = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, nullable=False)
    threat_level = db.Column(db.String(20), default='-')
    status = db.Column(db.String(20), nullable=False)

    # uselist=False 表示是一对一关系，这使得你可以从 UploadHistory 实例中直接访问关联的 BasicInfo 等的数据，反之亦然。如果想反之不然，删掉back_populates参数即可。
    basic_info = relationship("BasicInfo", uselist=False, back_populates="upload_history")
    pe_info = relationship("PEInfo", uselist=False, back_populates="upload_history")
    yara_match = relationship("YaraMatch", uselist=False, back_populates="upload_history")
    sigma_match = relationship("SigmaMatch", uselist=False, back_populates="upload_history")
    analyze_strings = relationship("AnalyzeStrings", uselist=False, back_populates="upload_history")

    def __repr__(self):
        return f'<UploadHistory {self.file_id}>'
    
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class BasicInfo(db.Model):
    __tablename__ = 'basic_info'
    # ondelete参数支持级联删除，根据upload_history中的情况，如果被删了，这边自动删除
    file_id = Column(Integer, ForeignKey('upload_history.file_id', ondelete='CASCADE'), primary_key=True)
    file_name = Column(String(255), nullable=False)
    file_size = Column(Float, nullable=False)
    file_type = Column(String(255))
    mime_type = Column(String(100))
    analyze_time = Column(DateTime)
    md5 = Column(String(32))
    sha1 = Column(String(40))
    sha256 = Column(String(64))
    upload_history = relationship("UploadHistory", back_populates="basic_info")
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class PEInfo(db.Model):
    __tablename__ = 'pe_info'
    file_id = Column(Integer, ForeignKey('upload_history.file_id', ondelete='CASCADE'), primary_key=True)
    machine_type = Column(String(10))
    timestamp = Column(DateTime)
    subsystem = Column(String(50))
    dll_characteristics = Column(String(50))
    sections = Column(Text) # JSON string or similar for storing complex structures
    imports = Column(Text)
    exports = Column(Text)

    upload_history = relationship("UploadHistory", back_populates="pe_info")
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class YaraMatch(db.Model):
    __tablename__ = 'yara_match'
    file_id = Column(Integer, ForeignKey('upload_history.file_id', ondelete='CASCADE'), primary_key=True)
    rule_name = Column(String(255))
    tags = Column(Text)
    strings = Column(Text)
    meta = Column(Text)

    upload_history = relationship("UploadHistory", back_populates="yara_match")
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class SigmaMatch(db.Model):
    __tablename__ = 'sigma_match'
    file_id = Column(Integer, ForeignKey('upload_history.file_id', ondelete='CASCADE'), primary_key=True)
    matches = Column(Text)

    upload_history = relationship("UploadHistory", back_populates="sigma_match")
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class AnalyzeStrings(db.Model):
    __tablename__ = 'analyze_strings'
    file_id = Column(Integer, ForeignKey('upload_history.file_id', ondelete='CASCADE'), primary_key=True)
    ascii_strings = Column(Text().with_variant(Text(length=4294967295), 'mysql'))  # 修改为长文本类型
    unicode_strings = Column(Text().with_variant(Text(length=4294967295), 'mysql'))

    upload_history = relationship("UploadHistory", back_populates="analyze_strings")
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


