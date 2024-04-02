
from flask import Blueprint, render_template

from . import qa_blueprint


@qa_blueprint.route('/qa', methods=['GET'])
def simple_respond():
    return render_template('qa.html')