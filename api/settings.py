import joblib
import json


APP = 'rpp-ai-first-person-fraud-co-ms'

# Load models
model_rappi = joblib.load('./data/model/model.joblib.dat')
imputer = joblib.load('./data/model/missing_imputer.joblib.dat')
explainer = joblib.load('./data/model/shap_explainer.joblib.dat')

t_file = open('./metrics/test_set_metrics.json')
threshold = json.load(t_file)["test_set"]["Threshold"]
t_file.close()

# Variables
with open('./data/variables/vars_raw.json', 'r') as file:
    vars_input = json.load(file)

MODEL_INPUT = vars_input.get("MODEL_INPUT")
FINAL_DATA = vars_input.get("FINAL_DATA")
CATEGORICAL_FEATURES = vars_input.get("CATEGORICAL_FEATURES")
NUMERICAL_FEATURES = vars_input.get("NUMERICAL_FEATURES")
BOOLEAN_FEATURES = vars_input.get("BOOLEAN_FEATURES")
CATEGORIES = vars_input.get("CATEGORIES")
OTHER = vars_input.get("OTHER")

VARIABLES_INPUT = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + \
    BOOLEAN_FEATURES
