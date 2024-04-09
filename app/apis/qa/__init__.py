import os

from flask import Blueprint

from qa_chain.pipeline import get_simple_rag_chain
from utils import ARXIVER_PATH

qa_blueprint = Blueprint('qa_api', __name__)
pipeline = get_simple_rag_chain(
    [ARXIVER_PATH / 'papers' / 'llm_apps' / 'mixtral_of_experts.pdf'], 
    'mixtral_of_experts')

from . import routes, events