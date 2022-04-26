import json
import pandas as pd
import joblib


class FraudModel:
    with open('./data/variables/vars_raw.json', 'r') as file:
        data = json.load(file)
    with open('./metrics/test_set_metrics.json', 'r') as t_file:
        threshold = json.load(t_file)["test_set"]['Threshold']

    model_rappi = joblib.load('./data/model/model.joblib.dat')
    imputer = joblib.load('./data/model/missing_imputer.joblib.dat')
    explainer = joblib.load('./data/model/shap_explainer.joblib.dat')

    CATEGORICAL_FEATURES = data.get("CATEGORICAL_FEATURES")
    NUMERICAL_FEATURES = data.get("NUMERICAL_FEATURES")
    BOOLEAN_FEATURES = data.get("BOOLEAN_FEATURES")
    CATEGORIES = data.get("CATEGORIES")
    OTHER = data.get("OTHER")

    batch_input_vars = CATEGORICAL_FEATURES + NUMERICAL_FEATURES \
        + BOOLEAN_FEATURES

    @staticmethod
    def preprocess_categories_batch(data: pd.DataFrame) -> pd.DataFrame:
        input_vars = set(data.columns)

        for feature in FraudModel.data.get("CATEGORICAL_FEATURES"):
            data[feature].fillna("MISSING", inplace=True)
        for key in FraudModel.data.get("CATEGORIES").keys():
            if key in input_vars:
                for category in FraudModel.data.get("CATEGORIES")[key]:
                    data[key + '_' + str(category)] = data[key].map(
                        lambda x: 1 if x == category else 0)
        for key in FraudModel.data.get("OTHER"):
            data[key + '_OTHER'] = data[key].map(
                lambda x: 1 if x not in
                FraudModel.data.get("CATEGORIES")[key] else 0)

        for key in FraudModel.data.get("CATEGORIES").keys():
            del data[key]

        return data

    @staticmethod
    def preprocess_numeric(dataset: pd.DataFrame) -> pd.DataFrame:

        for var in FraudModel.data.get("NUMERICAL_FEATURES"):
            dataset[var] = dataset[var].fillna(0)
            dataset[FraudModel.data.get("NUMERICAL_FEATURES")] = \
                FraudModel.imputer.transform(
                dataset[FraudModel.data.get("NUMERICAL_FEATURES")])
        return dataset

    @staticmethod
    def preprocess_boolean(dataset: pd.DataFrame) -> pd.DataFrame:

        for feature in FraudModel.data.get("BOOLEAN_FEATURES"):
            dataset[feature] = dataset[feature].astype(int)

        return dataset

    @staticmethod
    def get_feature_vector_batch(data: pd.DataFrame) -> pd.DataFrame:
        data = FraudModel.preprocess_categories_batch(data)
        X = data[FraudModel.data.get("FINAL_DATA")]
        X = FraudModel.preprocess_numeric(X)
        X = FraudModel.preprocess_boolean(X)
        return X

    @staticmethod
    def explain_score(X: pd.DataFrame) -> dict:
        individual_shaps = [round(float(x), 2)
                            for x in FraudModel.explainer.shap_values(X)[0]]
        dictionary = dict(zip(X.columns.tolist(), individual_shaps))

        return dictionary

    @staticmethod
    def calculate_score(X: pd.DataFrame) -> float:
        proba = FraudModel.model_rappi.predict_proba(X)[0][1]
        proba = round(float(proba), 6)

        return proba

    @staticmethod
    def get_batch_model_response(data: pd.DataFrame) -> pd.DataFrame:
        input_vars = data[FraudModel.batch_input_vars]
        vars = input_vars.copy()
        X = FraudModel.get_feature_vector_batch(input_vars)
        X = X[FraudModel.data.get("FINAL_DATA")]
        proba = FraudModel.model_rappi.predict_proba(X)[:, 1]
        shaps = pd.DataFrame(FraudModel.explainer.shap_values(X.values),
                             columns=input_vars.columns).round(2)
        shaps = pd.DataFrame({"explainability": shaps.to_dict("records")})
        X = pd.DataFrame({"variables": vars.to_dict("records"),
                          "threshold": FraudModel.threshold})

        return proba, X  # , shaps
