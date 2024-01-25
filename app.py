
from flask import Flask

from apis.qa import qa_blueprint


app = Flask(__name__)
app.register_blueprint(qa_blueprint, url_prefix='/qa')


if __name__ == '__main__':
    app.run('127.0.0.1', 8080, debug=True)
