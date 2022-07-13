import base64
import requests
import json


def upload_json_dataroom(data, path):
    print("Uploading to dataroom..")
    url = "http://dataroom.liris.cnrs.fr:8080/remote.php/webdav/" + path
    token = base64.b64encode("neptune:neptune".encode('ascii'))

    headers = {
        "Authorization": "Basic " + token.decode('ascii')
    }

    rsp = requests.put(url, data=json.dumps(data), headers=headers)

    print(rsp)
