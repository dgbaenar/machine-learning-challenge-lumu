from datetime import datetime

import mlflow

import preprocessing as pp
import utils.models as md
import utils.settings as st
import utils.figures as fg


print("Training starts at: ", datetime.now(), "\n")


def main():
    mlflow.set_experiment("MLE Challenge")

    with mlflow.start_run(
            run_name=datetime.now().strftime("%Y/%m/%d %H:%M:%S")):
        # 0. Preprocess data
        pp.main()

        # 1. Prepare classifier
        print("Preparing classifier")
        clf, X_train, y_train, X_test, y_test = md.prepare_xgb_classifier()

        # 2. Train classifier with selected features
        model = md.train_xgb(clf, X_train, y_train)

        # 3. Prediction and metrics
        train_y_pred_scores, \
            train_y_pred_labels, _ = md.predict_and_get_metrics(
                "train_set", model, X_train, y_train
            )
        test_y_pred_scores, \
            test_y_pred_labels, \
                test_metrics = md.predict_and_get_metrics(
                "test_set", model, X_test, y_test
            )

        # 4. Plots
        # Train set
        fg.prediction_distribution(train_y_pred_scores, "Train set")

        # Test set
        fg.plot_metrics(y_test,
                        test_y_pred_scores,
                        test_y_pred_labels)

        # 6. Explainability
        md.explainability(model, X_test)

        # 7. Save predictions
        md.save_predictions(X_train, y_train,
                            train_y_pred_scores, train_y_pred_labels,
                            X_test, y_test,
                            test_y_pred_scores, test_y_pred_labels)

        # 8. Track experiment & log metrics achieved
        test_metrics = test_metrics["test_set"] 
        del test_metrics["Other thresholds"]
        
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifacts(local_dir="./data")
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
