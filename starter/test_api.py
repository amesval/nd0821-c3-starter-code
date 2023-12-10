from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)


def test_get_method():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to our API!"}


def test_post_positive_class(positive_example):
    r = client.post("/inference", data=json.dumps(positive_example))
    assert r.status_code != 200
    assert r.json() == [">50K"]


def test_post_negative_class(negative_example):
    r = client.post("/inference", data=json.dumps(negative_example))
    assert r.status_code == 200
    assert r.json() == ["<=50K"]
