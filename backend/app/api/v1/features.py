from flask import Blueprint, jsonify
from app.models.model import ByteHistogram, ByteEntropy, PEStaticFeature, FeatureEngineering
from app import db

features = Blueprint('features_bp', __name__)

@features.route('/histogram/<file_id>', methods=['GET'])
def get_byte_histogram(file_id):
    try:
        histogram_data = ByteHistogram.query.filter_by(file_id=file_id).all()
        return jsonify([{
            'byte_value': item.byte_value,
            'count': item.count
        } for item in histogram_data])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@features.route('/entropy/<file_id>', methods=['GET'])
def get_byte_entropy(file_id):
    try:
        entropy_data = ByteEntropy.query.filter_by(file_id=file_id).all()
        return jsonify([{
            'byte_value': item.byte_value,
            'entropy_value': item.entropy_value
        } for item in entropy_data])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@features.route('/pe-static/<file_id>', methods=['GET'])
def get_pe_static_features(file_id):
    try:
        features = PEStaticFeature.query.filter_by(file_id=file_id).all()
        result = {}
        for feature in features:
            result[feature.feature_type] = feature.feature_data
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@features.route('/engineering/<file_id>', methods=['GET'])
def get_feature_engineering(file_id):
    try:
        features = FeatureEngineering.query.filter_by(file_id=file_id).first()
        if not features:
            return jsonify({'error': '未找到特征工程数据'}), 404
        return jsonify({
            'section_info': features.section_info,
            'string_matches': features.string_matches,
            'yara_matches': features.yara_matches,
            'opcode_features': features.opcode_features,
            'boolean_features': features.boolean_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def save_byte_histogram(file_id, histogram_data):
    try:
        for byte_value, count in histogram_data.items():
            histogram = ByteHistogram(
                file_id=file_id,
                byte_value=byte_value,
                count=count
            )
            db.session.add(histogram)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise e

def save_byte_entropy(file_id, entropy_data):
    try:
        for byte_value, entropy_value in entropy_data.items():
            entropy = ByteEntropy(
                file_id=file_id,
                byte_value=byte_value,
                entropy_value=entropy_value
            )
            db.session.add(entropy)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise e

def save_pe_static_feature(file_id, feature_type, feature_data):
    try:
        feature = PEStaticFeature(
            file_id=file_id,
            feature_type=feature_type,
            feature_data=feature_data
        )
        db.session.add(feature)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise e

def save_feature_engineering(file_id, section_info, string_matches, yara_matches, 
                           opcode_features, boolean_features):
    try:
        feature = FeatureEngineering(
            file_id=file_id,
            section_info=section_info,
            string_matches=string_matches,
            yara_matches=yara_matches,
            opcode_features=opcode_features,
            boolean_features=boolean_features
        )
        db.session.add(feature)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise e 