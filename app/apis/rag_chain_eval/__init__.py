from flask import Blueprint

rag_chain_eval_blueprint = Blueprint('rag_chain_eval_api', __name__)

from . import events, routes