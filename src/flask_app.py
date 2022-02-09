from abc import ABC, abstractmethod

from flask import Flask


class FlaskApp(ABC):

    def __init__(self, name):
        self.app = Flask(name)
        self.add_all_endpoints()

    def run(self, host='127.0.0.1', port=8000, debug=True):
        self.app.run(host=host, port=port, debug=debug)

    def add_endpoint(self, endpoint, endpoint_name, handler):
        self.app.add_url_rule(endpoint, endpoint_name, handler)

    @abstractmethod
    def add_all_endpoints(self):
        pass

