import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from shap import Explainer, plots, summary_plot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score)
from sklearn import set_config
from fairlearn.metrics import (
    MetricFrame,
    false_negative_rate,
    selection_rate,
    count,
)
from fairlearn.postprocessing import plot_threshold_optimizer
from datasets import prepare_test_train_datasets, resample_dataset, figures_test_train, load_dataset
from directories import generated, clean_dirs, clean_specific_dir, test_train_dir, unmitigated_dir, \
    mitigated_to_dir, mitigated_eg_dir, shap_dir
from mitigators import get_threshold_optimizer, get_exponentiated_gradient1
from models import coefficients_odds, roc_curve_lr, display_performance_hg, train_model_lr, train_model_hg
from settings import categorical_features

pd.set_option("display.float_format", "{:.3f}".format)
set_config(display="diagram")
sns.set()


# https://github.com/fairlearn/talks/blob/main/2022_pycon/pycon-2022-students.ipynb
# https://fairlearn.org/v0.10/auto_examples/plot_grid_search_census.html


def show_counts_sensitive_variables(df, show=False):
    plt.rcParams["figure.figsize"] = (20, 10)
    df["race"].value_counts().plot(kind='bar', rot=45, title='Patients, grouped by race, count.')
    plt.savefig(generated() + 'race_counted.png')
    if show:
        plt.show()
    df.gender.value_counts().plot(kind='bar', rot=45, title='Patients, grouped by gender, count.')
    plt.savefig(generated() + 'gender_counted.png')
    if show:
        plt.show()
    dfg = df.groupby(by=["race", "gender", "readmit_30_days"], observed=False).size()
    dfg.plot(kind='barh', title='Patients, grouped by race, gender and readmit_30_days, counted.')
    plt.savefig(generated() + 'patients_grouped_counted.png')
    if show:
        plt.show()
    plt.clf()



def shap():
    clean_specific_dir(shap_dir())
    random_seed = 445
    np.random.seed(random_seed)
    df = load_dataset()
    info(df)
    sensitive_features = ['race', 'gender']
    X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test = prepare_test_train_datasets(df, random_seed)
    X_train_bal, Y_train_bal, A_train_bal = resample_dataset(X_train, Y_train, A_train)

    Y_pred_proba, Y_pred, unmitigated_pipeline = train_model_lr(X_train_bal, Y_train_bal, X_test)
    model = unmitigated_pipeline.named_steps['logistic_regression']
    model_fitted = model.fit(X_train_bal, Y_train_bal)

    explainer = Explainer(model, X_train_bal, feature_names=X_train_bal.columns.to_list())
    shap_values = explainer(X_test)
    plots.waterfall(shap_values[0])
    summary_plot(shap_values, X_test, plot_type="bar")

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



    sensitive_features = ['race', 'gender']

    metrics_dict = {
        "selection_rate": selection_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": balanced_accuracy_score,
        "count": count
    }

    if show_pivot:
        pivot(df, False)

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
        metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred_postprocess, use_log_reg, True,False)

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


def groups_percentages(df, column_names):
    for col_name in column_names:
        count = df[col_name].value_counts()
        percentage = df[col_name].value_counts(normalize=True)
        df_ = pd.concat([count, percentage], axis=1, keys=('Count', 'Percentage'))
        print(col_name + ':', df_)
        df_.to_csv(generated() + col_name + '.csv')


def info(df):
    df.info()
    print(df.isna().sum())
    print(df.A1Cresult.unique)
    df = df.drop_duplicates(keep='first')
    print('number of duplicates:', df.duplicated(subset=None, keep='first').sum())


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


def metrics(metrics_dict, sensitive_features, Y_test, Y_pred, use_log_reg, use_treshold, unmitigated):
    if unmitigated:
        filename_part = 'unm_'
        dir = unmitigated_dir()
    elif use_treshold:
        if use_log_reg:
            filename_part = 'mit_to_lr_'
        else:
            filename_part = 'mit_to_hg_'
        dir = mitigated_to_dir()
    elif not use_treshold:
        if use_log_reg:
            filename_part = 'mit_eg_lr_'
        else:
            filename_part = 'mit_eg_hg_'
        dir = mitigated_eg_dir()
    #print("metrics_"+ filename_part)
    clean_specific_dir(dir)

    metricframe_ = MetricFrame(metrics=metrics_dict,
                                          y_true=Y_test,
                                          y_pred=Y_pred,
                                          sensitive_features=sensitive_features)

    metricframe_.by_group.round(2).to_csv(dir+ filename_part+"metrics.csv")

    metrics_aggregated = pd.DataFrame({'difference': metricframe_.difference(),
                            'ratio': metricframe_.ratio(),
                            'group_min': metricframe_.group_min(),
                            'group_max': metricframe_.group_max()
                            }).T
    metrics_aggregated.astype(float).round(2).to_csv(dir +filename_part +"metrics_agg.csv")

    #metrics_.to_excel(generated_dir(use_log_reg) + filename_part + 'sf_metrics.xlsx')
    #metrics_.to_pickle(generated_dir(use_log_reg) + filename_part + 'sf_metrics.pkl')

    metricframe_.by_group.plot.bar(subplots=True, layout=[2, 2], figsize=(12, 12),
                                              legend=False, rot=90, position=.5)
    plt.savefig(dir + filename_part + 'mf.png')
    plt.show()


def explore_eg_predictors(eg, X_test, Y_test, A_test):
    pass
    '''predictors = eg.predictors_
    print(predictors)

    sweep_preds = [clf.predict(X_test) for clf in predictors]
    balanced_error_sweep = [1 - balanced_accuracy_score(Y_test, Y_sweep) for Y_sweep in sweep_preds]
    fnr_diff_sweep = [false_negative_rate(Y_test, Y_sweep, sensitive_features=A_test).difference() for Y_sweep in
                      sweep_preds]'''
