from flask import jsonify, request, make_response, Blueprint, Flask

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['POST'])
def index():
    return "Welcome!"