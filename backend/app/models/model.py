from sqlalchemy import Column, Integer, String, Float, DateTime, BIGINT
from flask_sqlalchemy import SQLAlchemy
from app import db

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.password}')"


class UploadHistory(db.Model):
    __tablename__ = 'upload_history'
    id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    environment = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, nullable=False)
    threat_level = db.Column(db.String(20), default='low', nullable=False)
    status = db.Column(db.String(20), default='pending', nullable=False)

    def __repr__(self):
        return f'<UploadHistory {self.file_name}>'
