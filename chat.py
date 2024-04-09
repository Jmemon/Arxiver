
from dotenv import load_dotenv

import eventlet
eventlet.monkey_patch()

from app import socketio, create_qa_app
from utils import ARXIVER_PATH


load_dotenv(str(ARXIVER_PATH / '.env'))

if __name__ == '__main__':
    app = create_qa_app()

    socketio.run(app, '0.0.0.0', 8080, log_output=True)
    #socketio.run(app, '192.168.1.5', 8080, log_output=True)
    #app.run('0.0.0.0', 8080)  # to find IP address
