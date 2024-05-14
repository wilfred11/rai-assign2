from operator import add

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_curve,
    RocCurveDisplay)
from sklearn import set_config
from fairlearn.metrics import (
    MetricFrame,
    false_negative_rate,
    selection_rate,
    count,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity

from datasets import prepare_test_train_datasets, resample_dataset, figures_test_train
from directories import generated, clean_dirs, generated_dir, clean_specific_dir, test_train_dir, unmitigated_dir

pd.set_option("display.float_format", "{:.3f}".format)
set_config(display="diagram")
sns.set()


# https://github.com/fairlearn/talks/blob/main/2022_pycon/pycon-2022-students.ipynb
# https://fairlearn.org/v0.10/auto_examples/plot_grid_search_census.html

def load_dataset():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/fairlearn/talks/main/2021_scipy_tutorial/data/diabetic_preprocessed.csv")

    print(data.head())

    data.to_csv('./data/diabetic.csv', sep=';')

    data = data.drop(columns=[
        "discharge_disposition_id",
        "readmitted",
        #"readmit_30_days"
    ])

    data = delete_rows(data)
    #data["race_all"] = data["race"].copy()
    data["race"] = data["race"].replace({"Asian": "Other", "Hispanic": "Other"})
    data["diabetesMed"] = data["diabetesMed"].replace({"Yes": True, "No": False})

    # Show the values of all binary and categorical features
    categorical_values = {}
    for col in data:
        if col not in {'time_in_hospital', 'num_lab_procedures',
                       'num_procedures', 'num_medications', 'number_diagnoses'}:
            categorical_values[col] = pd.Series(data[col].value_counts().index.values)
    categorical_values_df = pd.DataFrame(categorical_values).fillna('')
    #categorical_values_df.T

    for col_name in categorical_features():
        data[col_name] = data[col_name].astype("category")
    return data


def delete_rows(df):
    df.drop(df[df['gender'] == 'Unknown/Invalid'].index, inplace=True)
    print('gender unique:', df.gender.unique())
    return df


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
        roc_curve_lr(Y_test, Y_pred_proba, Y_pred)
        if show_coefficients:
            coefficients(unmitigated_pipeline, X_test.columns)
            coefficients1(unmitigated_pipeline, X_test.columns)
    else:
        Y_pred, unmitigated_pipeline = train_model_hg(X_train_bal, Y_train_bal, X_test)
        display_performance_hg(Y_test, Y_pred)

    if show_metrics_before:
        clean_specific_dir(unmitigated_dir())
        metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred, df_test, use_log_reg, False, True)

    to = get_threshold_optimizer(unmitigated_pipeline)
    to.fit(X_train_bal, Y_train_bal, sensitive_features=A_train_bal)
    Y_pred_postprocess = to.predict(X_test, sensitive_features=A_test)

    if show_metrics_after:
        metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred_postprocess, df_test, use_log_reg, True,
                True)

    if use_log_reg:
        estimator = unmitigated_pipeline.named_steps['logistic_regression']

        #eg = get_exponentiated_gradient(estimator)
        eg = get_exponentiated_gradient1(estimator, random_seed)
        #** {'LogisticRegression__sample_weight': weights}
        #eg.fit(X_train_bal, Y_train_bal, )
        #eg.fit(X_train_bal, Y_train_bal,sensitive_features=A_train_bal, **{'LogisticRegression__sample_weight':sample_weight2})
        #kw = {'sensitive_features': A_train_bal, 'sample_weight': sample_weight2}
        print(unmitigated_pipeline.named_steps['logistic_regression'])
        eg.fit(X_train_bal, Y_train_bal, sensitive_features=A_train_bal)
        Y_pred_reductions = eg.predict(X_test, random_state=random_seed)

        if show_metrics_after:
            metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred_reductions, df_test, use_log_reg, False,
                    True)

        #explore_eg_predictors(eg, X_test, Y_test, A_test)


def graphs_test_train(A_train_bal):
    sns.countplot(x="race", data=A_train_bal)
    plt.title("Sensitive Attributes for Training Dataset")

def coefficients(unmitigated_pipeline, columns, show=False):
    coef_series = pd.Series(data=unmitigated_pipeline.named_steps["logistic_regression"].coef_[0], index=columns)
    coef_series.sort_values().plot.bar(figsize=(80, 20), legend=False, fontsize=20)
    plt.savefig(generated_dir(True) + 'lr_coef.png')
    if show:
        plt.show()
    #plt.clf()

def coefficients1(unmitigated_pipeline, columns, show=False):
    odds = np.exp(unmitigated_pipeline.named_steps["logistic_regression"].coef_[0])
    coefs = pd.DataFrame(odds, columns, columns=['coef']).sort_values(by='coef', ascending=False)
    coefs.plot.bar(figsize=(80, 40), legend=False, fontsize=15)
    plt.axhline(y=1, color='red', lw=.5)
    #plt.axhline(y=1, color='r', linestyle='-')
    #ax.axvline(1, color="red", linestyle="--", lw=2, label="")
    plt.savefig(generated_dir(True) + 'lr_coef1.png')
    if show:
        plt.show()



def roc_curve_lr(Y_test, Y_pred_proba, Y_pred, show=False):
    print("display performance")
    #fpr, tpr, threshold = roc_curve(Y_test, Y_pred_proba)
    #p = RocCurveDisplay(fpr=fpr, tpr=tpr, plot_chance_level= True)
    p = RocCurveDisplay.from_predictions(Y_test, Y_pred_proba, plot_chance_level=True)
    plt.plot([0, 1], [0, 1], 'k--', label='')
    p.plot()
    if show:
        plt.show()
    plt.savefig(generated_dir(True) + 'lr_roc_curve.png')
    plt.clf()
    print(balanced_accuracy_score(Y_test, Y_pred))


def display_performance_hg(Y_test, Y_pred, show=False):
    fpr, tpr, threshold = roc_curve(Y_test, Y_pred)
    p = RocCurveDisplay(fpr=fpr, tpr=tpr, plot_chance_level= True)
    plt.plot([0, 1], [0, 1], 'k--', label='')
    p.plot()

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


def numeric_and_binary_features():
    return ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses',
            'medicare', 'medicaid', 'had_emergency', 'had_inpatient_days', 'had_outpatient_days', 'readmit_binary',
            'diabetesMed']


def categorical_features():
    return ["race", "gender", "age", "admission_source_id", "medical_specialty", "primary_diagnosis", "max_glu_serum",
            "A1Cresult", "insulin", "change"]


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
    #summary(df).sort_values(by='Uniques', ascending=False)[:20]
    #print(summary(df).sort_values(by='Nulls', ascending=False))


def summary(df, pred=None):
    obs = df.shape[0]
    Types = df.dtypes
    Counts = df.apply(lambda x: x.count())
    Min = df.min()
    Max = df.max()
    Uniques = df.apply(lambda x: x.unique().shape[0])
    Nulls = df.apply(lambda x: x.isnull().sum())
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['Types', 'Counts', 'Uniques', 'Nulls', 'Min', 'Max']
        str = pd.concat([Types, Counts, Uniques, Nulls, Min, Max], axis=1, sort=True)

    str.columns = cols
    print('___________________________\nData Types:')
    print(str.Types.value_counts())
    print('___________________________')
    return str


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


def metrics(metrics_dict, sensitive_features, Y_test, Y_pred, df_test, use_log_reg, use_treshold, unmitigated):

    if unmitigated:
        filename_part = 'unm_'
    elif use_treshold:
        filename_part = 'mit_to_'
    elif not use_treshold:
        filename_part = 'mit_eg_'
    print("metrics_"+ filename_part)

    metricframe_unmitigated = MetricFrame(metrics=metrics_dict,
                                          y_true=Y_test,
                                          y_pred=Y_pred,
                                          sensitive_features=sensitive_features)

    # The disaggregated metrics are stored in a pandas Series mf1.by_group:

    print(metricframe_unmitigated.by_group)
    metricframe_unmitigated.by_group.round(2).to_csv(unmitigated_dir()+"unmitigated_selection_rate.csv")

    print(metricframe_unmitigated.difference())

    metrics = pd.DataFrame({'difference': metricframe_unmitigated.difference(),
                            'ratio': metricframe_unmitigated.ratio(),
                            'group_min': metricframe_unmitigated.group_min(),
                            'group_max': metricframe_unmitigated.group_max()
                            }).T

    print(metrics)
    metrics.to_excel(generated_dir(use_log_reg) + filename_part + 'sf_metrics.xlsx')
    metrics.to_pickle(generated_dir(use_log_reg) + filename_part + 'sf_metrics.pkl')

    metricframe_unmitigated.by_group.plot.bar(subplots=True, layout=[2, 2], figsize=(12, 12),
                                              legend=False, rot=90, position=.5)
    plt.savefig(generated_dir(use_log_reg) + filename_part + 'mf.png')
    plt.show()

    '''metricframe_unmitigated.by_group.plot(
        kind="bar",
        ylim=[0, 1],
        subplots=True,
        layout=[1, 4],
        legend=False,
        figsize=[12, 12],
        title="Show all metrics with assigned y-axis range",
    )

    plt.show()'''

    '''metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[3, 3],
        legend=False,
        figsize=[12, 8],
        title="Show all metrics",
    )'''


def get_threshold_optimizer(unmitigated_pipeline):
    postprocess_est = ThresholdOptimizer(
        estimator=unmitigated_pipeline,
        constraints="false_negative_rate_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method='predict_proba'
    )
    return postprocess_est


def get_exponentiated_gradient(unmitigated_pipeline):
    expgrad_est = ExponentiatedGradient(
        estimator=unmitigated_pipeline,
        constraints=TruePositiveRateParity(difference_bound=0.02),
        #sample_weight_name='sample_weight'
        sample_weight_name="logistic_regression__sample_weight",

    )
    return expgrad_est


def get_exponentiated_gradient1(unmitigated_pipeline, random_seed):
    expgrad_est = ExponentiatedGradient(
        estimator=LogisticRegression(max_iter=1000, random_state=random_seed),
        constraints=TruePositiveRateParity(difference_bound=0.02),
        sample_weight_name='sample_weight'

    )
    return expgrad_est


def explore_eg_predictors(eg, X_test, Y_test, A_test):
    pass
    '''predictors = eg.predictors_
    print(predictors)

    sweep_preds = [clf.predict(X_test) for clf in predictors]
    balanced_error_sweep = [1 - balanced_accuracy_score(Y_test, Y_sweep) for Y_sweep in sweep_preds]
    fnr_diff_sweep = [false_negative_rate(Y_test, Y_sweep, sensitive_features=A_test).difference() for Y_sweep in
                      sweep_preds]'''
