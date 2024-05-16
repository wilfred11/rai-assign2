import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from counts_metrics import show_counts_sensitive_variables, groups_percentages, metrics
from shap import Explainer, plots, summary_plot, LinearExplainer, initjs
from sklearn import set_config
from fairlearn.postprocessing import plot_threshold_optimizer
from datasets import prepare_test_train_datasets, resample_dataset, figures_test_train, load_dataset
from directories import generated, clean_dirs, clean_specific_dir, test_train_dir, unmitigated_dir, \
    shap_dir
from mitigators import get_threshold_optimizer, get_exponentiated_gradient1
from models import coefficients_odds, roc_curve_lr, display_performance_hg, train_model_lr, train_model_hg, \
    train_model_lr_
from settings import categorical_features, metrics_dict, sensitive_features



# https://github.com/fairlearn/talks/blob/main/2022_pycon/pycon-2022-students.ipynb
# https://fairlearn.org/v0.10/auto_examples/plot_grid_search_census.html


def shap(random_seed):
    clean_specific_dir(shap_dir())
    df = load_dataset()

    X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test = prepare_test_train_datasets(df, random_seed)
    X_train_bal, Y_train_bal, A_train_bal = resample_dataset(X_train, Y_train, A_train)

    unmitigated_pipeline = train_model_lr_(X_train_bal, Y_train_bal)
    #Y_pred_proba = unmitigated_pipeline.predict_proba(X_test)[:, 1]
    #print('X_test cols:', X_test.columns )
    #Y_test_pred = unmitigated_pipeline.predict(X_test)
    model = unmitigated_pipeline.named_steps['logistic_regression']

    model.fit(X_train_bal, Y_train_bal)

    Y_test_pred=model.predict(X_test)
    print('len ytestpred:', len(Y_test_pred))
    print('len ytest:', len(Y_test))
    #print('pred:', Y_test_pred.count(1))
    '''plt.scatter(Y_test, Y_test_pred)
    plt.plot()
    plt.show()'''

    plt.scatter(Y_test, Y_test_pred, )
    plt.xlabel('Actual Petal Width')
    plt.ylabel('Predicted Petal Width')
    plt.title('Actual vs Predicted Petal Width')
    plt.show()


    model = unmitigated_pipeline.named_steps['logistic_regression']

    explainer = LinearExplainer(model, X_train_bal, feature_names=X_train_bal.columns.to_list())
    shap_values = explainer(X_test)

    plt.subplots_adjust(left=0.46)
    plots.waterfall(shap_values[0], show=False, max_display=11)
    plt.savefig(shap_dir() + "shap_waterfall_0.png")
    plt.clf()

    #plt.subplots_adjust(right=0.895, left=0.422, wspace = 0.2  )
    summary_plot(shap_values, X_test, plot_type="bar", show=True)
    #plt.savefig(shap_dir() + "shap_summary.png")
    #plt.clf()

    #plt.subplots_adjust(left=0.16, right=1.97)
    plots.beeswarm(shap_values, max_display=20, show=True)
    #plt.figure().set_figwidth(10)
    #plt.figure(figsize=(10,30))
    #plt.savefig(shap_dir()+"shap_beeswarm.png")
    plt.clf()

    #plots.scatter(shap_values[:, "time_in_hospital"], color=shap_values)

    #plots.scatter(shap_values[:, "RM"], color=shap_values)




def medical(show_counts_sf, show_pivot, show_train_test, show_coefficients,
            show_metrics_before,
            show_metrics_after, use_log_reg):
    clean_dirs()
    random_seed = 445
    np.random.seed(random_seed)

    df = load_dataset()

    info(df)

    if show_counts_sf:
        show_counts_sensitive_variables(df, False)

    sensitive_features = sensitive_features()

    metrics_dict = metrics_dict()

    #if show_pivot:
    #    pivot(df, False)

    groups_percentages(df, ['gender', 'race', 'age'])

    X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test = prepare_test_train_datasets(df, random_seed)
    X_train_bal, Y_train_bal, A_train_bal = resample_dataset(X_train, Y_train, A_train)

    clean_specific_dir(test_train_dir())
    if show_train_test:
        figures_test_train(A_train_bal, Y_train_bal, A_test, Y_test)

    if use_log_reg:
        Y_pred_proba, Y_pred, unmitigated_pipeline = train_model_lr(X_train_bal, Y_train_bal, X_test)
        roc_curve_lr(Y_test, Y_pred_proba, False)
        if show_coefficients:
            #coefficients(unmitigated_pipeline, X_test.columns, False)
            coefficients_odds(unmitigated_pipeline, X_test.columns, False)
    else:
        Y_pred, unmitigated_pipeline = train_model_hg(X_train_bal, Y_train_bal, X_test)
        display_performance_hg(Y_test, Y_pred, False)

    if show_metrics_before:
        clean_specific_dir(unmitigated_dir())
        metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred, use_log_reg, False, True)

    to = get_threshold_optimizer(unmitigated_pipeline)
    to.fit(X_train_bal, Y_train_bal, sensitive_features=A_train_bal)
    Y_pred_postprocess = to.predict(X_test, sensitive_features=A_test)
    plot_threshold_optimizer(to, ax=None, show_plot=True)

    if show_metrics_after:
        metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred_postprocess, use_log_reg, True, False)

    if use_log_reg:
        estimator = unmitigated_pipeline.named_steps['logistic_regression']
        eg = get_exponentiated_gradient1(estimator, random_seed)
        print(unmitigated_pipeline.named_steps['logistic_regression'])
        eg.fit(X_train_bal, Y_train_bal, sensitive_features=A_train_bal)
        Y_pred_reductions = eg.predict(X_test, random_state=random_seed)

        if show_metrics_after:
            metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred_reductions, use_log_reg, False,
                    False)

        #explore_eg_predictors(eg, X_test, Y_test, A_test)

def do_lr(df, random_seed, do_train_test):
    X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test = prepare_test_train_datasets(df, random_seed)
    X_train_bal, Y_train_bal, A_train_bal = resample_dataset(X_train, Y_train, A_train)

    if do_train_test:
        clean_specific_dir(test_train_dir())
        figures_test_train(A_train_bal, Y_train_bal, A_test, Y_test)

    Y_pred_proba, Y_pred, unmitigated_pipeline = train_model_lr(X_train_bal, Y_train_bal, X_test)




    to = get_threshold_optimizer(unmitigated_pipeline)
    to.fit(X_train_bal, Y_train_bal, sensitive_features=A_train_bal)
    to.predict(X_test, sensitive_features=A_test)
    plot_threshold_optimizer(to, ax=None, show_plot=True)




def graphs_test_train(A_train_bal):
    sns.countplot(x="race", data=A_train_bal)
    plt.title("Sensitive Attributes for Training Dataset")


def denominalize(df):
    print('denominalize')
    df_ = df.copy()
    print('cols:', df.columns)
    df_ = df_.drop(columns=[
        #"race",
        #"race_all",
        #"discharge_disposition_id",
        #"readmitted",
        "readmit_binary",
        #"readmit_30_days"
    ])

    df_denom = pd.get_dummies(df_[categorical_features()])
    cols_to_be_filtered = [item for item in df_ if item not in categorical_features()]
    df_[cols_to_be_filtered].to_csv(generated() + 'data_numeric_bin.csv')
    df_denom.to_csv(generated() + 'data_denom.csv')
    cols = [df_denom, df_[cols_to_be_filtered]]
    df_expanded = pd.concat(cols, axis=1)
    df_expanded.to_csv(generated() + 'data_expanded.csv')
    return df_expanded


def info(df):
    df.info()
    print(df.isna().sum())
    print(df.A1Cresult.unique)
    print("dtypes:", df.dtypes)


def pivot(df, show=False):

    print("pivot")
    dfg = df.groupby(by=["race", "gender", "readmit_30_days"]).size()
    # A_train_bal[['race', 'gender']].value_counts().reset_index(name='count').plot(kind='barh', )
    dfg.plot(kind='barh', title='Patients, grouped by race, gender and readmit_30_days, counted.')
    plt.savefig(generated() + 'patients_grouped_counted.png')
    if show:
        plt.show()

    '''pv = np.round(pd.pivot_table(df, values='readmit_30_days',
                                 index=['age'],
                                 observed=False,
                                 columns=['gender', 'race'],
                                 aggfunc=np.count_nonzero,
                                 fill_value=0), 0)'''

    np.round(pd.pivot_table(df, values='readmit_30_days',
                            index=['gender', 'race'],
                            observed=False,
                            columns=['age'],
                            aggfunc=np.count_nonzero,
                            fill_value=0), 0).plot.barh(figsize=(10, 7), title='ean car price by make and '
                                                                               'number of doors')
    plt.savefig(generated() + 'pivot_.png')
    if show:
        plt.show()

    #plt.clf()

    #print(pv)


def explore_eg_predictors(eg, X_test, Y_test, A_test):
    pass
    '''predictors = eg.predictors_
    print(predictors)

    sweep_preds = [clf.predict(X_test) for clf in predictors]
    balanced_error_sweep = [1 - balanced_accuracy_score(Y_test, Y_sweep) for Y_sweep in sweep_preds]
    fnr_diff_sweep = [false_negative_rate(Y_test, Y_sweep, sensitive_features=A_test).difference() for Y_sweep in
                      sweep_preds]'''
