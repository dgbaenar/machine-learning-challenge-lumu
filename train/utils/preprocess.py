import ast
import datetime
import json
import joblib
import numpy as np
import pandas as pd
import re

from sklearn import model_selection, impute

import utils.settings as st


def prepare_dataset(raw_path, test_path, metric_col):
    # Read dataset
    data = pd.read_csv(raw_path, sep=";")
    data.columns = data.columns.str.upper()

    # Create features
    # Dia de la semana
    data["DIA_DE_LA_SEMANA"] = data["FECHA"].apply(
        lambda x: str(datetime.datetime.strptime(x, '%d/%m/%Y').strftime("%A")))
    # Monto
    data["MONTO"] = data["MONTO"].apply(
        lambda x: float(x.replace(',', '.'))).apply(lambda x: float(x))
    # Descuento
    data["DCTO"] = data["DCTO"].apply(
        lambda x: float(x.replace(',', '.'))).apply(lambda x: float(x))
    # Cashback
    data["CASHBACK"] = data["CASHBACK"].apply(
        lambda x: float(x.replace(',', '.'))).apply(lambda x: float(x))
    # Desagregar columna dispositivo
    data["DISPOSITIVO"] = data['DISPOSITIVO'].apply(
        lambda x: x.replace(";", ",")).apply(ast.literal_eval)
    data = pd.concat([data.drop(['DISPOSITIVO'], axis=1),
                      pd.json_normalize(data["DISPOSITIVO"])], axis=1)
    # Remover caracteres especiales en tipo_tc
    data["TIPO_TC"] = data["TIPO_TC"].apply(
        lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))
    # Cambiar type de variable dependiente
    data[st.METRIC_NAME] = data[st.METRIC_NAME].astype(int)
    data.columns = data.columns.str.upper()

    # Separate backtesting dataset and raw dataset
    data["FECHA"] = data["FECHA"].apply(
        lambda x: str(datetime.datetime.strptime(x, '%d/%m/%Y')))
    data["FECHA"] = pd.to_datetime(data['FECHA'])
    backtesting_set = data[data["FECHA"] >=
                           (data["FECHA"].max() - datetime.timedelta(days=7))]
    data = data[data["FECHA"] <
                (data["FECHA"].max() - datetime.timedelta(days=7))]

    # Save raw and bactesting set processed
    data.to_csv("./data/raw_processed.csv", index=False)
    backtesting_set.to_csv("./data/backtesting.csv", index=False)

    # Select variables
    data = data[st.CATEGORICAL_FEATURES + st.NUMERICAL_FEATURES +
                st.BOOLEAN_FEATURES + [metric_col]].copy()
    # Sample to test
    to_test = data.sample(1)
    to_test = to_test.T.to_dict()[to_test.index[0]]
    with open(test_path, 'w') as outfile:
        json.dump(to_test, outfile, indent=4)

    return data


def preprocess_variables(data: pd.DataFrame, imputer_path):
    # Preprocess numerical
    if st.NUMERICAL_FEATURES:
        # Imput Missing values with mean
        imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        data[st.NUMERICAL_FEATURES] = imputer.fit_transform(
            data[st.NUMERICAL_FEATURES])
        joblib.dump(imputer, imputer_path)

    # Preprocess categorical
    if st.CATEGORICAL_FEATURES:
        # Imput Missing Values
        data[st.CATEGORICAL_FEATURES] = data[st.CATEGORICAL_FEATURES].fillna(
            "MISSING")
        # Convert Case
        data[st.CATEGORICAL_FEATURES] = data[st.CATEGORICAL_FEATURES].apply(
            lambda x: x.astype(str).str.upper())

        for feature in st.CATEGORICAL_FEATURES:
            data[feature] = pd.Categorical(
                values=data[feature], categories=st.CATEGORIES[feature])
            if feature in st.OTHER:
                data[feature] = data[feature].cat.add_categories(
                    "OTHER")
                data[feature] = data[feature].fillna("OTHER")

        # Dummies
        categorical = pd.get_dummies(data[st.CATEGORICAL_FEATURES])
        data = data.join(categorical)
        data = data.drop(st.CATEGORICAL_FEATURES, axis=1)

    # Preprocess boolean
    if st.BOOLEAN_FEATURES:
        # Imput Missing values
        for feature in st.BOOLEAN_FEATURES:
            data[feature] = data[feature].astype(int)

    return data


def split_dataset(data, test_size, random_state, metric_col,
                  train_dataset_path, test_dataset_path):
    # Split datasets
    print('Splitting...')
    train, test = model_selection.train_test_split(
        data, test_size=test_size, random_state=random_state)
    # Downsampling
    print('Rebalancing...')
    train = pd.concat([train[train[metric_col] == 0].sample(train[metric_col].value_counts().loc[1], random_state=st.RANDOM_STATE),
                       train[train[metric_col] == 1]])
    # Save datasets
    train.to_csv(train_dataset_path, index=False)
    test.to_csv(test_dataset_path, index=False)
