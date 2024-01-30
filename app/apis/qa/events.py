

from llm import get_llama

llm = get_llama()


def stream_llm_output(prompt):
    for chunk in llm.stream(prompt):
        yield chunk


@socketio.on('message', namespace='/qa/simple')
def handle_message(message):
    # Assuming message contains the prompt for the LLM
    prompt = message.get('prompt')
    if prompt:
        for chunk in stream_llm_output(prompt):
            emit('response', {'data': chunk}, namespace='/qa/simple')