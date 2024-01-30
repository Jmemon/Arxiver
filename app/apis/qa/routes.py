
from flask import Blueprint

qa_blueprint = Blueprint('qa_api', __name__)


@qa_blueprint.route('/simple', methods=['GET'])
def simple_respond():
    return render_template('qa/simple.html')