from flask import Flask, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)


class Disease(Resource):
    def post(self):
        json_data = request.get_json()

        return {'result': 'Success'}, 200  # return data with 200 OK


api.add_resource(Disease, '/disease')  # '/users' is our entry point

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444, debug=True)
