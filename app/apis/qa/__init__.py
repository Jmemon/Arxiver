import os

from flask import Blueprint

from qa_chain.pipeline import get_simple_rag_chain
from utils import ARXIVER_PATH


qa_blueprint = Blueprint('qa_api', __name__)

if os.environ['ARXIVER_CLOUD'] == 'True':
    """
    TODO:
    Programmatically launch lambda cloud gpu instance and load rag pipeline onto it, writing to stdout what's going on as it happens
    For now I'm just going to launch an instance and put it on
    """
    from .cloud import routes, events
else:
    pipeline = get_simple_rag_chain(
        [ARXIVER_PATH / 'papers' / 'mixtral_of_experts.pdf'], 
        'mixtral_of_experts')

    from .local import routes, events