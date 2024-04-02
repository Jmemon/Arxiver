
import eventlet
from flask_socketio import emit

from app import socketio
from app.apis.rag_chain_eval import pipeline, interm_values


@socketio.on('message', namespace='/rag_chain_eval')
def handle_message(message):
    # Assuming message contains the prompt for the LLM
    prompt = message.get('prompt')
    if prompt:
        response = ''
        print(prompt)
        for chunk in pipeline.stream(prompt):
            emit('response', {'data': chunk}, namespace='/rag_chain_eval') 
            response += chunk
            exit()

            # This forces the server to take control for a brief second so the LLM isn't taking up
            # all processing power. Otherwise all tokens are emitted after they're all generated, defeating the
            # point of streaming
            eventlet.sleep(0)
            
        emit('intermediate_chain_values', {'intermediate_values': interm_values.intermediate_values}, namespace='/rag_chain_eval')
        
