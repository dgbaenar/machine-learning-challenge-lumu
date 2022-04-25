import logging
import json

from flask import Flask, request, render_template

from api.exceptions import ModelParamException, ModelNotFoundException
from api.utils import get_model_response


# Initialize App
app = Flask(__name__)


@app.route('/', methods=['GET'])
def health():
    return render_template("index.html")


@app.route('/eda', methods=['GET'])
def eda():
    return render_template("eda.html")


@app.route('/calculate', methods=['POST'])
def calculate_risk():
    try:
        feature_dict = request.form["modelinput"]
        feature_dict = json.loads(feature_dict)
    except ValueError as e:
        return {'error': str(e)}, 422

    if not feature_dict or feature_dict == "":
        return {
            'error': 'Body is empty.'
        }, 400
    try:
        response = get_model_response(feature_dict)
    except ModelParamException as e:
        return {'error': str(e)}, 400
    except ModelNotFoundException as e:
        return {'error': str(e)}, 400
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 400

    return response, 200


if __name__ == '__main__':
    logging.info('Server started')
    app.run(host='0.0.0.0', port=8000, use_reloader=True)
