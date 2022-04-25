import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import joblib
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
import shap
import xgboost as xgb

import utils.settings as st


def prepare_xgb_classifier(train_dataset_path=st.TRAIN_DATASET_PATH,
                           test_dataset_path=st.TEST_DATASET_PATH,
                           metric_col=st.METRIC_NAME):
    # Initialize XGB model
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=1337
    )
    # Read Train/Test
    train = pd.read_csv(train_dataset_path)
    test = pd.read_csv(test_dataset_path)
    # Train
    y_train = train.pop(metric_col)
    X_train = train[st.FINAL_DATA]

    y_test = test.pop(metric_col)
    X_test = test[st.FINAL_DATA]

    return clf, X_train, y_train, X_test, y_test


def rfecv(clf, X_train, y_train, cv=5):
    # Initialize RFECV
    models = RFECV(clf, cv=cv)
    model = models.fit(X_train, y_train)

    print("Optimal number of features :", model.n_features_)

    features_ranking = list(model.ranking_)
    # Select variables
    ordered_variables = sorted(list(zip(st.FINAL_DATA, features_ranking)),
                               key=lambda x: x[1])

    print("Selected variables:\n", ordered_variables[:model.n_features_])
    filtered_variables = [i for i, _ in
                          ordered_variables[:model.n_features_]]
    # Save final features
    model_input = [[i for j in filtered_variables if i in j]
                   for i in st.VARIABLES_INPUT if i]
    categorical_features = [[i for j in filtered_variables if i in j]
                            for i in st.CATEGORICAL_FEATURES if i]
    numerical_features = [[i for j in filtered_variables if i in j]
                          for i in st.NUMERICAL_FEATURES if i]
    if st.BOOLEAN_FEATURES:
        boolean_features = [[i for j in filtered_variables if i in j]
                            for i in st.BOOLEAN_FEATURES if i]

    with open(st.VARS_FILTERED_PATH, 'w') as fd:
        json.dump(
            {
                "MODEL_INPUT": [i[0] for i
                                in model_input if len(i) > 0],
                "FINAL_DATA": filtered_variables,
                "CATEGORICAL_FEATURES": [i[0] for i
                                         in categorical_features
                                         if len(i) > 0],
                "NUMERICAL_FEATURES": [i[0] for i
                                       in numerical_features if len(i) > 0],
                "BOOLEAN_FEATURES": [],
                "RFECV_RANKING": str(ordered_variables)
            },
            fd, indent=4
        )

    X_train = X_train[filtered_variables]
    X_test = X_test[filtered_variables]

    return X_train, X_test


def train_xgb(clf, X_train, y_train):
    # Initialize Random Search with CV
    rsearch = RandomizedSearchCV(
        estimator=clf,
        param_distributions=st.DISTRIBUTIONS,
        random_state=st.RANDOM_STATE,
        scoring='f1',
        cv=5,
        verbose=2,
        n_iter=20
    )
    # Train model
    print('Training...\n')
    models = rsearch.fit(X_train, y_train)
    model = models.best_estimator_
    # Export model
    print('Exporting model...')
    joblib.dump(model, st.MODEL_PATH)

    return model


def predict_and_get_metrics(set, model, X_set, y_set,
                            optimize='f2', beta=3):
    print("===========================================================")
    print("=====================", set, "=====================")
    print("===========================================================")
    # Predict scores
    y_pred_scores = model.predict_proba(X_set)[:, 1]

    # Calculate metrics and thresholds
    precisions, recalls, thresholds = metrics.precision_recall_curve(
        y_set, y_pred_scores)

    # Optimize f1 or f2 score
    if optimize == 'f1':
        # F1 formula (matrix)
        optimize_metric = (2 * (recalls * precisions)) / (recalls + precisions)

    elif optimize == 'f2':
        # F2 formula (matrix)
        optimize_metric = (1 + beta ** 2) * (recalls * precisions) / \
            ((precisions * (beta ** 2)) + recalls)

    threshold = float(thresholds[np.argmax(optimize_metric)])
    precision = precisions[np.argmax(optimize_metric)]
    recall = recalls[np.argmax(optimize_metric)]
    roc_auc = metrics.roc_auc_score(y_set, y_pred_scores)
    y_pred_labels = (y_pred_scores >= threshold).astype(int)
    metrics_dict = {
        f"{set}": {f"Threshold": round(threshold, 3),
                   f"Precision": round(precision, 3),
                   f"Recall": round(recall, 3),
                   f"ROC AUC": round(roc_auc, 3),
                #    f"{optimize}": round(optimize_metric[np.argmax(optimize_metric)], 3),
                   "Other thresholds": {cut_point: "" for cut_point in st.CUT_POINTS}
                   }
    }

    print(f"Suggested Threshold: {threshold} | Precision: {precision} \
            | Recall: {recall}")
    print(metrics.confusion_matrix(y_set, y_pred_labels))

    # Generate metrics for other thresholds
    for cut_point in st.CUT_POINTS:
        precision_ = round(precisions[np.argwhere(thresholds >= cut_point)[0]][0], 3)
        recall_ = round(recalls[np.argwhere(thresholds >= cut_point)[0]][0], 3)
        optimize_metric_ = round(optimize_metric[np.argwhere(
            thresholds >= cut_point)[0]][0], 3)

        metrics_dict[f"{set}"]["Other thresholds"][cut_point] = f"{optimize}:\
              {optimize_metric_} | Precision: {precision_} \ | Recall: {recall_}"

        print(f"Other: Threshold: {cut_point} | {optimize}: {optimize_metric_} \
                Precision: {precision_} \ | Recall: {recall_}")

    # Save metrics
    with open(st.METRICS_PATH + f"{set}_metrics.json", 'w') as fd:
        json.dump(
            metrics_dict,
            fd, indent=4
        )

    return y_pred_scores, y_pred_labels, metrics_dict


def explainability(model, X_test):
    # Create explainer file
    print('Explainability...')
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, st.EXPLAINER_PATH)

    # Plot explainability
    shap_values = explainer(X_test)

    # Summarize the effects of all the features
    shap.plots.beeswarm(shap_values, plot_size=(15, 8),
                        max_display=20, show=False)
    # plt.subplots_adjust(left=0.3)
    plt.savefig(st.IMAGES_PATH + 'shap.png')
    plt.close()


def save_predictions(X_train, y_train,
                     train_y_pred_scores, train_y_pred_labels,
                     X_test, y_test,
                     test_y_pred_scores, test_y_pred_labels):
    # Train dataset
    X_train['REAL_LABEL'] = y_train
    X_train['PREDICTED_FRAUD_SCORE'] = train_y_pred_scores
    X_train['PREDICTED_FRAUD_LABEL'] = train_y_pred_labels

    # Test dataset
    X_test['REAL_LABEL'] = y_test
    X_test['PREDICTED_FRAUD_SCORE'] = test_y_pred_scores
    X_test['PREDICTED_FRAUD_LABEL'] = test_y_pred_labels

    # Save datasets
    X_train.to_csv("./data/train_scored.csv", index=False)
    X_test.to_csv("./data/test_scored.csv", index=False)
