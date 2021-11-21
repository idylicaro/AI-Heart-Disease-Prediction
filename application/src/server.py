import json
import random

import pandas as pd
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from src.main import evaluate
import os

app = Flask(__name__)
CORS(app)
api = Api(app)


class Disease(Resource):
    def post(self):
        json_data = request.get_json()
        cwd = os.getcwd()  # Get the current working directory(cwd)
        json_path = random.randint(1, 1000)
        json_path = cwd+'/jsons/'+str(json_path)+'.json'
        with open(json_path, "w") as outfile:
            json.dump(json_data, outfile)
        # call evaluate in main.py with json
        result = evaluate(json_path, mlp_model_path=cwd+'/mlp_model.sav', heart_csv_path=cwd+'/heart.csv')
        if os.path.exists(json_path):
            os.remove(json_path)
        return {'result': result}, 200  # return data with 200 OK


api.add_resource(Disease, '/disease')  # '/users' is our entry point

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444, debug=True)
