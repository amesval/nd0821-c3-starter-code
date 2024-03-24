# Salary classification for Census Income dataset

In this project we use demographic information to predict if a person has an anual income that exceeds $50k per year. This project is based on census data (https://archive.ics.uci.edu/dataset/20/census+income).
The model is called from a FastAPI app that runs in a remote server (https://render.com/).

This project is part of the ML DevOps Engineer Nanodegree (Udacity).

# Setup environment

1. Clone repository:
```
git clone https://github.com/amesval/salary_classification.git
```

2. Create a conda environment:
```
conda create --name <ENV_NAME> python=3.8.19
```

3. Activate environment
```
conda activate <ENV_NAME>
```

4. Install requirements file
```
pip install -r src/requirements.txt
```

5. Switch to src/ directory:
```
cd src
```

# Data

The dataset location of the Census dataset (https://archive.ics.uci.edu/dataset/20/census+income) can be found in *src/data/census.csv*. A clean version of the dataset can be found here *src/data/clean_census.csv*.

# Exploratory Data Analysis

A dataset exploration can be found in the *src/eda.ipynb* notebook. A clean version of the dataset (*src/data/clean_census.csv*) is generated here.

# Training

The income classification is perform by a Random Forest Classification model. To train a new model with the *src/data/clean_census.csv* just run:
```
python -m ml.train_model
```

The metrics of the train model are saved in *src/model/metrics_report_model.txt*.

When a training is execute, it is possible to obtained the metrics for a particular feature once the model is trained. By default the *workclass* feature is selected. You can change the feature by using the *feature_to_slice* argument:
```
python -m ml.train_model --feature_to_slice <FEATURE_NAME>
```

The metrics for the selected feature can be found in *src/model/slice_output.txt*.

# Model

The trained model is saved as *src/model/model.pkl*. The model output could be >50K (Positive class), <=50K (Negative class).

The metrics of the current model are in *src/model/metrics_report_model.txt*.

You can check more details of the trained model in the *src/model_card_template.md*, including, ethical considerations, caveats and recommendations.

# API Creation

An API was created using FastAPI (https://fastapi.tiangolo.com/). The API implements:
* GET method on the root (giving a welcome message).
* POST method that performs the model inference.

The API uses type hinting, and a Pydantic model to ingest the body from POST. You can check the implementation in the *src/main.py* file.

Two examples are provided to test for a positive and negative class. Feel free to inspect both files:
1) *src/api_body_example_neg.json*
2) *src/api_body_example_pos.json*

# Software test cases

To test the API functionality, *src/test_api.py* defines 3 cases to validate the POST and GET methods.

In *src/ml/test_functions.py* you can find multiple test cases to verify that the training, the generation of artifacts, and the inference pipeline works as expected.

To run all the test, just run from the terminal
```
pytest -v 
```

# CI/CD with GitHub Actions

For every new commit to the *master branch*, a GitHub Action workflow is trigger. This build, test, and deploy our solution in a *Render* server (https://render.com/). 

The workflow install python dependencies, run tests, and lint checks before deploying the solution. For more details, feel free to review *./github/workflows/python-app.yml*. Note that our GitHub Actions is running *pytest* and *flake8*.



# API Deployment

You can test our solution which deploys the API in a *Render* server (https://salary-prediction-api-s2x0.onrender.com).

The *src/main.py* defines the FastAPI app, the GET and POST method, and their endpoints.

*src/post_to_api.py* is used to call the POST method from Python.

Note: To deploy your API in a new server, sign up for a *Render* account (https://render.com/) and follow the instructions to link your GitHub repository to the server.

# GET method (server solution)

- To call the GET method just open the server in the browser (https://salary-prediction-api-s2x0.onrender.com/).
- You can also try the GET method from the API documentation: https://salary-prediction-api-s2x0.onrender.com/docs#.
- To test with Postman (https://www.postman.com/) just add the endpoint url (https://salary-prediction-api-s2x0.onrender.com/) to the GET method.

# POST method (server solution)

1) To make predictions from Python switch to */src/* directory
2) Run:
```
python post_to_api.py --endpoint_url https://salary-prediction-api-s2x0.onrender.com/inference --input_filename <FILENAME>
```
Note: You can replace \<FILENAME\> with *api_body_example_neg.json* or *api_body_example_pos.json*

For *Postman* or the API docs (https://salary-prediction-api-s2x0.onrender.com/docs#) you can just copy your example into the *body*.

Note: If you want to test with Postman, make sure to use the correct endpoint (https://salary-prediction-api-s2x0.onrender.com/inference).

# Run API locally

1) Move to /src/ directory
2) Run:
```
uvicorn main:app --reload
```
- GET Method endpoint: http://127.0.0.1:8000 (or http://localhost:8000/)
- POST Method endpoint: http://127.0.0.1:8000/inference. 

The way to call both methods is analogous as using the *Render* server.

For more information, feel free to check https://fastapi.tiangolo.com/tutorial/first-steps/.

# License

[License](LICENSE.txt)

