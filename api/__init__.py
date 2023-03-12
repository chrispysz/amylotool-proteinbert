from flask import Flask
from flask_cors import CORS
import tensorflow as tf

model = tf.keras.models.load_model('./models/ProteinBERT')

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['SECRET_KEY'] = 'KHRl4KwU6KkD3AdMaG2jkANKvJLxPAQ8'
    app.config['TIMEOUT'] = None
    
    from .predictionsAPI import predictionsAPI

    app.register_blueprint(predictionsAPI, url_prefix='/predict')

    
    return app