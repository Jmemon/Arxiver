
import eventlet
eventlet.monkey_patch()

from app import socketio, create_rag_chain_eval_app

app = create_rag_chain_eval_app()


if __name__ == '__main__':
    socketio.run(app, '0.0.0.0', 5000, log_output=True)
    #socketio.run(app, '192.168.1.5', 8080, log_output=True)
    #app.run('0.0.0.0', 8080)  # to find IP address
