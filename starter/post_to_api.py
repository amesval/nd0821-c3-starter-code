import argparse
import json
import requests

parser = argparse.ArgumentParser(description='Inference with API')

default_url = 'https://salary-prediction-api-s2x0.onrender.com/inference'
parser.add_argument('--endpoint_url',
                    type=str,
                    help='Endpoint to make prediction',
                    default=default_url)
parser.add_argument('--input_filename',
                    type=str,
                    help="Name of the filename to make inference",
                    default='./api_body_example_neg.json'
                    )
args = parser.parse_args()

# Read an example (Body)
with open(args.input_filename, 'r') as f:
    data = json.load(f)
data = json.dumps(data)

# Make inference in the API
status = requests.post(args.endpoint_url, data=data)
response = status.json()
print(f"status: {status}    response: {response}")
