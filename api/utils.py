import pandas as pd

from api.exceptions import ModelParamException
import api.settings as st


def preprocess_categories(feature_dict, input_vars):
    # Fill Missing
    for feature in st.CATEGORICAL_FEATURES:
        if feature_dict[feature] is None:
            feature_dict[feature] = "MISSING"
    # Fill Categories
    for key in st.CATEGORIES.keys():
        if key in input_vars:
            for category in st.CATEGORIES[key]:
                if feature_dict[key] == category:
                    feature_dict[key + '_' + str(category)] = 1
                else:
                    feature_dict[key + '_' + str(category)] = 0
    # Get OTHER
    for key in st.OTHER:
        if feature_dict[key] not in st.CATEGORIES[key]:
            feature_dict[key + '_OTHER'] = 1
        else:
            feature_dict[key + '_OTHER'] = 0
    # Remove key
    for key in st.CATEGORIES.keys():
        del feature_dict[key]
    
    # Fill boolean
    for feature in st.BOOLEAN_FEATURES:
        feature_dict[feature] = int(feature_dict[feature])

    return feature_dict


def preprocess_numeric(dataset):
    # Fill with 0
    for var in st.NUMERICAL_FEATURES:
        dataset[var] = dataset[var].fillna(0)

    # Fill Missing
    dataset[st.NUMERICAL_FEATURES] = st.imputer.transform(
        dataset[st.NUMERICAL_FEATURES])

    return dataset


def get_feature_vector(feature_dict):
    # Validate input
    input_vars = set(feature_dict.keys())

    if input_vars != st.MODEL_INPUT:
        # Check if there is missing param
        missing = set(st.MODEL_INPUT) - set(input_vars)
        if len(missing) > 0:
            raise ModelParamException(f'Missing params: {list(missing)}')

    # Preprocess categories
    feature_dict = preprocess_categories(feature_dict, input_vars)

    # Get model
    X = pd.DataFrame(feature_dict, index=[0])[st.FINAL_DATA]

    # Preprocces numerics
    X = preprocess_numeric(X)

    model = st.model_rappi

    return X, model


def explain_score(X, explainer):
    individual_shaps = [round(float(x), 2)
                        for x in explainer.shap_values(X)[0]]
    dictionary = dict(zip(X.columns.tolist(), individual_shaps))

    return dictionary


def calculate_score(X, model):
    proba = model.predict_proba(X)[0][1]
    proba = round(float(proba), 6)

    return proba


def get_model_response(json_data):
    X, model = get_feature_vector(json_data)
    probability = calculate_score(X, model)
    explain = explain_score(X, st.explainer)
    if probability >= st.threshold:
        fraud = 1
        label = 'Fraud'
    else:
        fraud = 0
        label = 'Not fraud'
    return {
        'status': 'ok',
        'score': probability,
        'label': label,
        'fraud': fraud,
        'version': 'v1.0.0',
        'explain': explain
    }
