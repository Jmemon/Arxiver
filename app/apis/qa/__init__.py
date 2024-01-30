from flask import Blueprint

qa_blueprint = Blueprint('qa_api', __name__)

from . import routes, events