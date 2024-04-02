from flask import Blueprint

qa_blueprint = Blueprint('qa_api', __name__)

from qa_chain.pipeline import get_simple_rag_chain
from utils import ARXIVER_PATH

pipeline = get_simple_rag_chain(
    [ARXIVER_PATH / 'papers' / 'mixtral_of_experts.pdf'], 
    'mixtral_of_experts')

from . import routes, events