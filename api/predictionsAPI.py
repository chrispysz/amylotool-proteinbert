import uuid
from flask import Blueprint, request, jsonify
from .window_predict import predict_window


predictionsAPI = Blueprint('predictionsAPI', __name__)


@predictionsAPI.route('/full', methods=['POST'])
def predictFull():

    try:
        sequence = request.json['sequence']
        if (sequence == "ping"):
            return jsonify(results = "Service reached")
        result = predict_window(sequence)
        return jsonify(
            results=result
        )
    except Exception as e:
        return f"An Error Occurred: {e}"
