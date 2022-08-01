from flask import Flask
from route import home_page, getPrediction
def create_app():
    app = Flask(__name__)
    app.add_url_rule('/', '/', home_page)
    app.add_url_rule('/getPrediction/', 'getPrediction', getPrediction)

    return app