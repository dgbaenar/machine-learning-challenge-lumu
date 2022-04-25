# Machine Learning Engineer Challenge

## UCI Default of Credit Card Clients Dataset

The goal of this project is to automate the prediction of fraudulent credit card transacttions, giving a score to them from 0 to 1, being closer to 1 riskiers. The current repository uses a XGBoost model with a grid search approach to find optimal hyperparameters.

## How to reproduce the experiment?

Start by pulling the repository.

Then, create a virtual environment using python 3.8 and install the requirements following the next steps (you need to have installed Python 3.8):

```console
virtualenv venv --python=python3.8
```
```console
source venv/bin/activate
```
```console
pip install -r requirements.txt
```

Located in the root of the repository (db-website), execute the following python files in order:

```console
python train/peprocessing.py
```
```console
python train/eda.py
```
```console
python train/train.py
```

The previous commands will preprocess the raw dataset, create the train and test sets, generate the exploratory data analysis and train the model.

## How to run the model?

### Option 1:

Use the website that I shared to you by email (https://jnge3ehtyv.us-east-1.awsapprunner.com/). The features used in the model are the ones located in the file "./data/variables/vars_raw.json" (MODEL_INPUT). Once the model receive that input, transform the variables to the variables in "FINAL_DATA" to make the predictions.

In the file located at "./test/test.json", there is an example of the body that the model receives. It is important to keep in mind that if any of the variables listed "MODEL_INPUT" is missing or if the type is not the same as indicated in the vars_raw.json file, the service will return an error.

### Option 2:

After creating the virtual environment, install the libraries and run:

Then run:

```console
python app.py
```

This command will rise the web app on the port 8000 of your computer.

### Option 3:

Build and run the docker container:

```console
docker build . -t db-website 
```
```console
docker run --rm -it -p 8000:8000 db-website
```

## How to deploy and how is deployed?

Use the Dockerfile to deploy the container in your cloud of preference. Currently, the model is deployed in a cluster of AWS App Runner, which by default create a Docker container with Python 3.8, copy all the files that are not in the dockerignore, install the requirements, test the health endpoint ("\") and deploy the container.

There are currently three endpoints.

* "/": Health: ensures that the app is deployed and renders the index.html as home of the website
* "/eda": Exploratory Data Analysis: renders the wda.html generated using sweetviz.
* "/calculate": Uses the json input to return the explainability of the model using shap, and at the end, the model label, score and version. The threshold used is the one that optimized the F2 score (recall) because the goal is to catch more fraudster transactions sacrifizing a bit the precision.
## Insights and how good the model is

According to the shap values located in "./metrics/img/shap.png", it can be noted the most important variables to detect fraud are: the "cashback", "amount", "linea_tc", "interes_tc", "hora", "device_score" and "dcto".

However, the metrics obtained by the test and train set are not very hopeful. As you can see in the "./metrics/test_set_metrics.json" the precision and recall are too low for the suggested threshold, with a tendency to increase the recall as the threshold decreases. In the case of the "./metrics/train_set_metrics.json", the metrics are high, giving us a signal of overfitting.

## Next steps

According to the above, for the next steps, I would establish that the following activities should be:

* To reduce the variance of the test set, we could increase the train dataset, improve the imbalanced dataset by implementing SMOTE, implement regularization techniques

* Normalize variables and implement feature selection techniques such as Recursive Feature Elimination with Cross Validation.

* Implement mlflow With Airflow (or kubeflow) for model registry, experiment tracking and monitoring.

