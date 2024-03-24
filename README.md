# Salary classification for Census Income dataset

In this project we use demographic information to predict if a person has an anual income that exceeds $50k per year. This project is based on census data (https://archive.ics.uci.edu/dataset/20/census+income).
The model is called from a FastAPI app that runs in a remote server (https://render.com/).

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
pip install -r requirements.txt
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

An API was created using FastAPI (https://fastapi.tiangolo.com/). The API implement:
* GET method on the root giving a welcome message.
* POST that does model inference.

The API uses type hinting, and a Pydantic model to ingest the body from POST. You can check the implementation in the *main.py* file.

Two examples are provided to test for a positive and negative class. Feel free to inspect both files:
1) api_body_example_neg.json
2) api_body_example_pos.json

# Software test cases

To test the API functionality, the *test_api.py* defines 3 cases to validate the POST and GET methods.

In the *starter/ml/test_functions.py* you can find multiple test cases to verify that the training, the generation of artifacts, and the inference pipeline works as expected.

To run all the test, just run from the terminal
```
pytest -v 
```

# CI/CD with GitHub Actions

For every new commit to the *master branch*, a GitHub Action workflow is trigger. This build, test, and deploy our solution in a *Render* server (https://render.com/). 

The workflow install python dependencies, run tests, and lint checks before deploying the solution. For more details, feel free to review *./github/workflows/python-app.yml*. Note that our GitHub Actions is running *pytest* and *flake8*.



# API Deployment

You can test our solution which deploys the API in a *Render* server.

Note: To deploy your API in a server, sign up for a *Render* account (https://render.com/),
and follow the instructions to link your GitHub repository to the server.


* Write a script that uses the requests module to do one POST on your live API.

# License

[License](LICENSE.txt)

- Server: https://salary-prediction-api-s2x0.onrender.com/ (How to use GET method from the browser. Just Open the browser) live-get
- FastAPI: https://salary-prediction-api-s2x0.onrender.com/docs# (How to use GET method from the browser using FastAPI)
- Postman:
- Python:


3) Creamos dos ejemplos de prueba:
    a) api_body_example_neg.json
    b) api_body_example_pos.json
4) test_api.py define los test cases que se usan en el unit test
3) Hacemos predicciones con el modelo mediante diferentes enfoques:
    1) Python (utilizando el script de post_to_api.py):
        se utilizan los dos ejemplos del paso 3
    2) Postman:
        - iniciamos la API
        - hacemos la inferencia en http://localhost:8000/inference

    - live_post.png (muestra el resultado de usar post desde python)
    - postman_example.png (muestra el resultado  desde la API de python)


post_to_api.py llama el metodo post de la API (local y remota)
test_set.csv es uno de los datasets a utilizar (que es el test_set2.csv)

##########################################################
Hay tres formas para llamar los metodos get y post:

To start a local app we call:
    - uvicorn main:app --reload
    if we go to http://127.0.0.1:8000 or http://localhost:8000/
    we can observe the get method there or from server https://salary-prediction-api-s2x0.onrender.com/

1) Python (To test the post method only)
    - using localhost (python post_to_api.py --endpoint_url http://127.0.0.1:8000/inference --input_filename ./api_body_example_neg.json)
    - using server (python post_to_api.py --endpoint_url https://salary-prediction-api-s2x0.onrender.com/inference --input_filename ./api_body_example_neg.json)
2) Postman:
    - using localhost (uvicorn main:app --reload.. more info https://fastapi.tiangolo.com/tutorial/first-steps/)
    - using server (already working in the render server)
3) FastAPI (in the browser):
    - Either for server of for local API
        - server: (https://salary-prediction-api-s2x0.onrender.com/docs)
        - localhost: (http://127.0.0.1:8000/docs or http://localhost:8000.docs)
    - call the get method (say hello)
    - call the post method (inference)
##########################################################

- starter
    - conftest.py
    - test_api.py
    - sanitycheck.py (run sanity check to meet udacity criteria)
    - eda.ipynb (crea las estadisticas y limpia el dataset)
    - main.py (script to call the FastAPI. it also defines the get and post method and their endpoints)
    - post_to_api.py (llama el metodo post para ejecutarlo en python)
    - setup.py (included at the beginning, check what it does)
    - starter
        - conftest.py
        - test_functions.py
        - train_model.py (???)
        - ml
            - data.py (???)
            - model.py (???)

# Note: The server is running the app

# NOTE: UNTIL NOW WE NEED TO EXPLAIN
1) [Done]Run test cases
2) Run API inferences in:
    - Python
    - FastAPI
    - Postman
    (Either for server and local app)
3) Run test cases
4) [Done] Explain how to train the model
5) [Done] Explain EDA process
6) [Done] Explain data?? (explain how to clean data)
7) [Done] Explain model?? (explain model metrics??)
8) [Done] Explain CI/CD with GitHub Actions
9) [Done] Explain screenshots? Not needed
10) [Done]Explain model card
11) [Done] Add License

# TODO: After all
1) Rename repo
2) Change structure of the repo
3) Remove the sanity check
3) Change GitHub Actions "cd starter"