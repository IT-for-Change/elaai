import os
import json
import requests
from typing import Any, Dict, List


class Client:

    def __init__(self) -> None:
        # Load app config frpm environment
        app_config = os.environ.get("APP_CONFIG", "{}")
        try:
            self.config = json.loads(app_config)
        except json.JSONDecodeError as e:
            raise ValueError("APP_CONFIG is not valid JSON") from e

        self.remote_host = self.config.get("ela_api_host", {})
        self.remote_port = self.config.get("ela_api_port", {})
        self.api_token = self.config.get("ela_api_token", {})
        self.get_api = self.config.get("ela_get_api", {})
        self.put_api = self.config.get("ela_put_api", {})
        self.remote_get_api_url = f'http://{self.remote_host}:{self.remote_port}/api/method/{self.get_api}'
        self.remote_put_api_url = f'http://{self.remote_host}:{self.remote_port}/api/method/{self.put_api}'
        # "http://elaexplore.localhost:8081/api/method/ela.ela.doctype.activity.activity.get_submissions"

        # if not self.base_url:
        #   raise ValueError("api.base_url must be set in APP_CONFIG")

    def get_data(self, activity_id, operation):
        headers = {
            "Authorization": f"token {self.api_token}"
        }
        params = {"activity_eid": activity_id, "operation": operation}
        response = requests.get(self.remote_get_api_url,
                                params, headers=headers)
        response.raise_for_status()
        return response

    def put_data(self, outputs, operation):
        headers = {
            "Authorization": f"token {self.api_token}"
        }
        params = {"outputs": outputs, "operation": operation}
        print(params)
        response = requests.post(self.remote_put_api_url,
                                 json=params, headers=headers)
        response.raise_for_status()
        print({response.text})

    def send_file(self, file_path):

        file_name = os.path.basename(file_path)
        headers = {
            "Authorization": f"token {self.api_token}"
        }
        data = {
            "is_private": 0
        }
        files = {
            'file': (file_name, open(file_path, 'rb'), 'application/octet-stream')
        }

        url = "http://elaexplore.localhost:8081/api/method/upload_file"
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        data = response.json()
        file_id = data['message']['name']
        return file_id
