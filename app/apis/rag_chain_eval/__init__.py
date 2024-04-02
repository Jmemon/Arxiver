from flask import Blueprint

rag_chain_eval_blueprint = Blueprint('rag_chain_eval_api', __name__)

from qa_chain.custom_callbacks import IntermediateValuesCallback
from qa_chain.pipeline import get_simple_rag_chain
from utils import ARXIVER_PATH

interm_values = IntermediateValuesCallback()
pipeline = get_simple_rag_chain(
    [ARXIVER_PATH / 'papers' / 'mixtral_of_experts.pdf'], 
    'mixtral_of_experts', )
    #callbacks=[interm_values])

from . import events, routes