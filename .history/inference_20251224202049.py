import requests
import json

url = "http://localhost:5000/invocations"
data = {
    "columns": ["feature1", "feature2", "feature3"],
    "data": [[1,2,3], [4,5,6], [7,8,9]]
}

response = requests.post(url, json=data)
print("Response:", response.json())