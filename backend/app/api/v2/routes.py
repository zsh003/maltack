from flask import jsonify, Blueprint
from .controllers import get_hello_v2

api_v2 = Blueprint('api_v2_bp', __name__)

@api_v2.route('/hello')
def hello_v2():
    return jsonify(get_hello_v2())