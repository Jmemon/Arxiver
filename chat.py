
import eventlet
eventlet.monkey_patch()

from app import socketio, create_app

app = create_app()


if __name__ == '__main__':
    socketio.run(app, '127.0.0.1', 8080, log_output=True)
