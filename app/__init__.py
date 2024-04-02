
from flask import Flask
from flask_socketio import SocketIO

from langchain_core.runnables import Runnable

from qa_chain.pipeline import get_simple_rag_chain
from utils import ARXIVER_PATH

socketio = SocketIO()


def create_qa_app():
    from app.apis.qa import qa_blueprint
    
    app = Flask(__name__)
    app.register_blueprint(qa_blueprint)

    socketio.init_app(app)
    return app


def create_rag_chain_eval_app():
    from app.apis.rag_chain_eval import rag_chain_eval_blueprint

    app = Flask(__name__)
    app.register_blueprint(rag_chain_eval_blueprint)

    socketio.init_app(app)
    return app
