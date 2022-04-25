from datetime import datetime

import utils.models as md
import utils.settings as st
import utils.figures as fg


print("Training starts at: ", datetime.now(), "\n")


def main():
    # 1. Prepare classifier
    print("Preparing classifier")
    clf, X_train, y_train, X_test, y_test = md.prepare_xgb_classifier()

    # # 2. Train model with Recursive Feature Elimination with CV
    # print('Fitting model with Recursive Feature Elimination with CV...\n')
    # X_train, X_test = md.rfecv(clf, X_train, y_train)

    # 3. Train classifier with selected features
    model = md.train_xgb(clf, X_train, y_train)

    # 4. Prediction and metrics
    train_y_pred_scores, \
        train_y_pred_labels = md.predict_and_get_metrics(
            "train_set", model, X_train, y_train
        )
    test_y_pred_scores, \
        test_y_pred_labels = md.predict_and_get_metrics(
            "test_set", model, X_test, y_test
        )

    # 5. Plots
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


if __name__ == "__main__":
    main()
