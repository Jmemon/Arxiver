
from flask import render_template, jsonify

from . import rag_chain_eval_blueprint
from utils import ARXIVER_PATH


@rag_chain_eval_blueprint.route('/rag_chain_eval')
def rag_chain_eval():
    return render_template('rag_chain_eval.html')


@rag_chain_eval_blueprint.route('/graph')
def graph():
    from qa_chain.pipeline import get_simple_rag_chain

    return jsonify(get_simple_rag_chain(
                        [ARXIVER_PATH / 'papers' / 'llm_apps' / 'mixtral_of_experts.pdf'], 
                        'mixtral_of_experts'
                    ).get_graph().to_json())