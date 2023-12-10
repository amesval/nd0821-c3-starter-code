import pytest
import json

@pytest.fixture(scope='session')
def positive_example():
    with open('./api_body_example_pos.json', 'r') as f:
        data = json.load(f)
    return data

@pytest.fixture(scope='session')
def negative_example():
    with open('./api_body_example_neg.json', 'r') as f:
        data = json.load(f)
    return data