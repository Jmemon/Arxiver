
from flask import request, Blueprint, Response, stream_with_context, render_template
from langchain.schema import HumanMessage

from llm import get_llm


qa_blueprint = Blueprint('qa_api', __name__)

llm = get_llm()


"""
For all routes /qa will be pre-pended when we register the blueprint in app.py
"""


@qa_blueprint.route('/simple', methods=['GET'])
def simple_respond():
    prompt = request.args.get('prompt')
    if request.headers.get('accept') == 'text/event-stream':
        return Response(stream_with_context(llm.stream(prompt)), content_type='text/event-stream')
    else:
        return render_template('qa/simple.html')
