
import eventlet
from flask_socketio import emit

from app import socketio
from llm import get_llama, llama_path

llm = get_llama(llama_path)


@socketio.on('message', namespace='/qa')
def handle_message(message):
    # Assuming message contains the prompt for the LLM
    prompt = message.get('prompt')
    if prompt:
        for chunk in llm.stream(prompt):
            emit('response', {'data': chunk}, namespace='/qa')

            # This forces the server to take control for a brief second so the LLM isn't taking up
            # all processing power. Otherwise all tokens are emitted after they're all generated, defeating the
            # point of streaming
            eventlet.sleep(0)
