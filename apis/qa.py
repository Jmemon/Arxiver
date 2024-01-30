
import asyncio
import io
import contextlib

from flask import request, Blueprint, Response, stream_with_context, render_template

from llm import get_llama


qa_blueprint = Blueprint('qa_api', __name__)

llm = get_llama()


"""
For all routes /qa will be pre-pended when we register the blueprint in app.py
"""


def stream_llm_output(prompt):
    for chunk in llm.stream(prompt):
        yield chunk

@qa_blueprint.route('/simple', methods=['GET'])
def simple_respond():
    if request.headers.get('accept') == 'text/event-stream':
        prompt = request.args.get('prompt')
        return Response(
            stream_with_context(stream_llm_output(prompt)), 
            status=200,
            content_type='text/event-stream')
    else:
        return render_template('qa/simple.html')
