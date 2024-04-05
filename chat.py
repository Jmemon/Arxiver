
import argparse
import os

import eventlet
eventlet.monkey_patch()

from app import socketio, create_qa_app

parser = argparse.ArgumentParser(
    prog='Arxiver',
    description='Talk about a reseach topic that interests you!',
)
parser.add_argument('--cloud', action='store_true', default=False)

#app = create_qa_app()

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['ARXIVER_CLOUD'] = f'{args.cloud}'

    app = create_qa_app()

    socketio.run(app, '0.0.0.0', 8080, log_output=True)
    #socketio.run(app, '192.168.1.5', 8080, log_output=True)
    #app.run('0.0.0.0', 8080)  # to find IP address
