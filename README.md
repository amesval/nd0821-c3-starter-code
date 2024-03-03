# Salary classification for Census Income dataset

In this project we use demographic information to predict if a person has an anual income that exceeds $50k per year, based on census data (https://archive.ics.uci.edu/dataset/20/census+income).
The model is called from a FastAPI app that runs in a remote server (https://render.com/).

# Setup environment

1. Clone repository:
```
git clone https://github.com/amesval/nd0821-c3-starter-code.git
```

2. Create a conda environment:
```
conda create --name <ENV_NAME> python=3.8
```

3. Activate environment
```
conda activate <ENV_NAME>
```

4. Install requirements file (move code to the other folder??)
```
pip install -r requirements.txt
```

# Data

TODO: Dataset location in the repository

# EDA

TODO: Explain EDA and were is located

# Training

TODO: 

## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

En el folder data estan los datasets:
    - census.csv
    - clean_census.csv (version limpia del census)

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.

# Training

# Test?

# License


Pasos:

1) Entrenamos unos modelos de ML
2) Guardamos el modelo y generamos unas estadisticas y el dataset limpio mediante el eda.ipynb
3) Creamos dos ejemplos de prueba:
    a) api_body_example_neg.json
    b) api_body_example_pos.json
3) Creamos un conftest.py que define los ejemplos a usar en el unit test con pytest (se utilizan los dos ejemplos de prueba)
4) test_api.py define los test cases que se usan en el unit test
3) Hacemos predicciones con el modelo mediante diferentes enfoques:
    1) Python (utilizando el script de post_to_api.py):
        se utilizan los dos ejemplos del paso 3
    2) Postman:
        - iniciamos la API
        - hacemos la inferencia en http://localhost:8000/inference
    

En el folder data estan los datasets:
    - census.csv
    - clean_census.csv (version limpia del census)
En el folder de model estan:
    - reporte de metricas: metrics_report_model.txt (tambien esta el metrics_report_model2.txt pero no se usa??? estaba probando la api?)
    - model.pkl (y model2.pkl que no se usa... tambien esta un rf_model.pkl)
    - slice_output.txt (el cual nos da las metricas por clase)

En el folder de screenshots:
    - continuos_deployment.png (imagen de muestra de github actions)
    - continuous_deployment2.png (muestra que se hace un deployment por medio de github actions, el cual se ejecuta cuando hago push o pull request)
    - continuous_integration.png (muestra que el pipeline ejecutado por github actions se ejecuta correctamente cuando se hace push.. al ejecutar esto se deploya al nuevo server,
    es decir tenemos un CI/CD?)
    - endpoint_logs_status.png (muestra los logs de nuestro server y de que ya esta arriba listo para usarse)
    - example_get.png (muestra como usar el metodo get desde el browser con FastAPI [https://salary-prediction-api-s2x0.onrender.com/docs#]) : 
            
        - para esto solo es necesario abrir el browser: https://salary-prediction-api-s2x0.onrender.com/
    - example_post.png (muestra el ejemplo que se usará con el metodo post desde el browser con FastAPI)
    - example_post_response.png (muestra el resultado de usar post desde el browser con FastAPI)
    - live_get.png (muestra el resultado de usar get en el browser)
    - live_post.png (muestra el resultado de usar post desde python)
    - postman_example.png (muestra el resultado  desde la API de python)
    - sanity_check.png (muestra el resultado del unit test en python)

El model_card_template.md es una carta del modelo

para que sirve el setup.py?

requirements.txt contiene el ambiente a crear

sanitycheck.py Corre el unit test en python

post_to_api.py llama el metodo post de la API (local y remota)

test_api.py define las tres pruebas unitarias que corre el sanitycheck?

test_set.csv es uno de los datasets a utilizar (que es el test_set2.csv)


BORRAR:
    - dvc_on_heroku_instructions.md


Necesito describir la salida del modelo >50K, <=50K
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

# To run the sanity check: It doesn't run the unit test.. it just to verify
# that we have the 3 test cases


# To run all the unit test just move inside ./starter and execute pytest (or pytest .).. its better to use pytest -v
is going to executre the 8 test cases (3 for the ./test_api.py and 5 ./test_functions.py)


# TODO EXPLAIN HOW TO TRAIN THE MODEL



# NOTE: UNTIL NOW WE NEED TO EXPLAIN
1) Run test cases
2) Run API inferences in:
    - Python
    - FastAPI
    - Postman
    (Either for server and local app)
3) Run test cases
4) Explain how to train the model
5) Explain EDA process
6) Explain data?? (explain how to clean data)
7) Explain model?? (explain model metrics??)
8) Explain CI/CD with GitHub Actions
9) Explain screenshots
10) Explain model card


# TODO: After all
1) Rename repo
2) Change structure of the repo