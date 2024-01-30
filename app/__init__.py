
from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO()


def create_app():
    from app.apis.qa import qa_blueprint
    
    app = Flask(__name__)
    app.register_blueprint(qa_blueprint)

    socketio.init_app(app)
    return app