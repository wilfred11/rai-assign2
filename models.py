import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_curve, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from directories import generated_dir


def coefficients(unmitigated_pipeline, columns, show=False):
    coef_series = pd.Series(data=unmitigated_pipeline.named_steps["logistic_regression"].coef_[0], index=columns)
    coef_series.sort_values().plot.bar(figsize=(80, 20), legend=False, fontsize=20)
    plt.savefig(generated_dir(True) + 'lr_coef.png')
    if show:
        plt.show()
    #plt.clf()


def coefficients_odds(unmitigated_pipeline, columns, show=False):
    odds = np.exp(unmitigated_pipeline.named_steps["logistic_regression"].coef_[0])
    coefs = pd.DataFrame(odds, columns, columns=['coef']).sort_values(by='coef', ascending=False)
    coefs.plot.bar(figsize=(80, 40), legend=False, fontsize=15)
    plt.axhline(y=1, color='red', lw=.5)
    plt.savefig(generated_dir(True) + 'lr_coef1.png')
    if show:
        plt.show()


def roc_curve_lr(Y_test, Y_pred_proba, show=False):
    print("display performance")
    #fpr, tpr, threshold = roc_curve(Y_test, Y_pred_proba)
    #p = RocCurveDisplay(fpr=fpr, tpr=tpr, plot_chance_level= True)
    p = RocCurveDisplay.from_predictions(Y_test, Y_pred_proba, plot_chance_level=True)
    plt.plot([0, 1], [0, 1], 'k--', label='')
    #p.plot()
    if show:
        plt.show()
    plt.savefig(generated_dir(True) + 'lr_roc_curve.png')
    plt.clf()


def display_performance_hg(Y_test, Y_pred, show=False):
    fpr, tpr, threshold = roc_curve(Y_test, Y_pred)
    p = RocCurveDisplay(fpr=fpr, tpr=tpr, plot_chance_level= True)
    plt.plot([0, 1], [0, 1], 'k--', label='')
    #p.plot()

    if show:
        plt.show()
    plt.savefig(generated_dir(False) + 'hg_roc_curve.png')

    print(balanced_accuracy_score(Y_test, Y_pred))


def train_model_lr(X_train_bal, Y_train_bal, X_test):
    unmitigated_pipeline = Pipeline(steps=[
        ("preprocessing", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=5000))

    ])

    unmitigated_pipeline.fit(X_train_bal, Y_train_bal)
    Y_pred_proba = unmitigated_pipeline.predict_proba(X_test)[:, 1]
    Y_pred = unmitigated_pipeline.predict(X_test)
    return Y_pred_proba, Y_pred, unmitigated_pipeline


def train_model_hg(X_train_bal, Y_train_bal, X_test):
    unmitigated_pipeline = Pipeline(steps=[
        ("preprocessing", StandardScaler()),
        ("hist_gradient_boosting_classifier", HistGradientBoostingClassifier(max_iter=1000))
    ])

    unmitigated_pipeline.fit(X_train_bal, Y_train_bal)

    #Y_pred_proba = unmitigated_pipeline.predict_proba(X_test)[:, 1]
    Y_pred = unmitigated_pipeline.predict(X_test)
    return Y_pred, unmitigated_pipeline
