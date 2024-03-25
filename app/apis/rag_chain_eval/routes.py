
from flask import render_template, jsonify

from . import rag_chain_eval_blueprint, pipeline
from utils import ARXIVER_PATH


@rag_chain_eval_blueprint.route('/rag_chain_eval')
def rag_chain_eval():
    return render_template('rag_chain_eval.html')


@rag_chain_eval_blueprint.route('/graph')
def graph():
    from qa_chain.pipeline import get_simple_rag_chain

    return jsonify(pipeline.get_graph().to_json())