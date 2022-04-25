import json

from scipy.stats import uniform, randint


# Options
DATASET_PATH = "./data/raw.csv"
TRAIN_DATASET_PATH = "./data/train.csv"
TEST_DATASET_PATH = "./data/test.csv"
TEST_SAMPLE_PATH = "./tests/test.json"
VARS_FILTERED_PATH = "./data/variables/vars_filtered.json"
IMPUTER_PATH = "./data/model/missing_imputer.joblib.dat"
EXPLAINER_PATH = "./data/model/shap_explainer.joblib.dat"

IMAGES_PATH = "./metrics/img/"
METRICS_PATH = "./metrics/"
MODEL_PATH = "./data/model/model.joblib.dat"
SCALER_PATH = "./data/model/scaler.joblib.dat"
METRIC_NAME = 'FRAUDE'
TEST_SIZE = 0.15
RANDOM_STATE = 23
CUT_POINTS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97]
COLORS = ['#1A3252', '#EB5434', '#377A7B']

# Distributions for Random search
DISTRIBUTIONS = {
    'max_depth': randint(low=3, high=31),
    'learning_rate': uniform(loc=0.1, scale=0.4),
    'gamma': uniform(loc=0.001, scale=0.998),
    'min_child_weight': uniform(loc=0.5, scale=4.5),
    'max_delta_step': uniform(loc=0.5, scale=2),
    'subsample': uniform(loc=0.5, scale=0.499),
    'colsample_bytree': uniform(loc=0.5, scale=0.499),
    'colsample_bylevel': uniform(loc=0.5, scale=0.499),
    'reg_alpha': uniform(loc=0.001, scale=0.998),
    'reg_lambda': uniform(loc=0.001, scale=0.998),
    'n_estimators': randint(low=20, high=300),
}

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
