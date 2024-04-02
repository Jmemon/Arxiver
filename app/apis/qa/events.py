
import eventlet
from flask_socketio import emit

from app import socketio
from qa_chain.models import mistral7b
from app.apis.qa import pipeline

stop_flag = False


@socketio.on('message', namespace='/qa')
def handle_message(message):
    global stop_flag

    # Assuming message contains the prompt for the LLM
    prompt = message.get('prompt')
    if prompt:
        for chunk in pipeline.stream(prompt):
            emit('response', {'data': chunk}, namespace='/qa')

            # This forces the server to take control for a brief second so the LLM isn't taking up
            # all processing power. Otherwise all tokens are emitted after they're all generated, defeating the
            # point of streaming
            eventlet.sleep(0)
            if stop_flag:
                break
        
        stop_flag = False


@socketio.on('stop', namespace='/qa')
def stop_llm_response():
    global stop_flag
    stop_flag = True
