import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay)
from sklearn import set_config
from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, TruePositiveRateParity
from datetime import date

pd.set_option("display.float_format", "{:.3f}".format)
set_config(display="diagram")
sns.set()


def generated_dir(use_log_reg):
    path = './generated/'
    if use_log_reg:
        return path + 'lr/'
    else:
        return path + 'hg/'


# https://github.com/fairlearn/talks/blob/main/2022_pycon/pycon-2022-students.ipynb
# https://fairlearn.org/v0.10/auto_examples/plot_grid_search_census.html

def load_dataset():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/fairlearn/talks/main/2021_scipy_tutorial/data/diabetic_preprocessed.csv")

    print(data.head())

    # Show the values of all binary and categorical features
    categorical_values = {}
    for col in data:
        if col not in {'time_in_hospital', 'num_lab_procedures',
                       'num_procedures', 'num_medications', 'number_diagnoses'}:
            categorical_values[col] = pd.Series(data[col].value_counts().index.values)
    categorical_values_df = pd.DataFrame(categorical_values).fillna('')
    categorical_values_df.T

    categorical_features = [
        "race",
        "gender",
        "age",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "primary_diagnosis",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
        "readmitted"
    ]
    print('un:', data.age.unique())

    for col_name in categorical_features:
        data[col_name] = data[col_name].astype("category")
    return data


def medical(show_predictive_validity, show_pivot, show_train_test, show_coefficients, show_metrics_before,
            show_metrics_after, use_log_reg):
    random_seed = 445
    np.random.seed(random_seed)

    df = load_dataset()

    info(df)
    sensitive_features = ['race', 'gender']
    metrics_dict = {
        "selection_rate": selection_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": balanced_accuracy_score,
        "count": count
    }

    df["race"].value_counts().plot(kind='bar', rot=45)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    sns.countplot(x='race', hue='readmit_30_days', data=df, ax=ax)
    plt.show()
    #sns.countplot(x='race', hue='salary >50k', data=df, ax=ax[1])
    #sns.kdeplot(x='age', hue='salary >50k', data=df_visu, ax=ax[2])

    #groups=df.groupby(['gender', 'race', 'age', 'readmit_30_days'])

    #to_be_grouped = df[["gender", "race", "age", "readmit_30_days"]].copy()
    #to_be_grouped.groupby(['race', 'gender', 'age']).readmit_30_days.value_counts().unstack(3).plot.barh()
    #plt.show()

    if show_pivot:
        pivot(df)

    # drop gender group Unknown/Invalid
    df = df.copy().query("gender != 'Unknown/Invalid'")

    # retain the original race as race_all, and merge Asian+Hispanic+Other
    df["race_all"] = df["race"].copy()
    df["race"] = df["race"].replace({"Asian": "Other", "Hispanic": "Other"})
    #df["race"] = df["race"].rename_categories({"Asian": "Other", "Hispanic": "Other"})

    if show_predictive_validity:
        pairwise_correlation(df)
        predictive_validity(df)

    print(df["readmit_30_days"].value_counts())  # counts
    print(df["readmit_30_days"].value_counts(normalize=True))  # frequencies

    sns.barplot(x="readmit_30_days", y="race", data=df, errorbar=('ci', 95))
    plt.savefig('./generated/' + 'bp_readmit_race.png')
    plt.show()

    sns.pointplot(y="medicaid", x="race", data=df, linestyle='none')
    plt.savefig('./generated/' + 'pp_medicaid_race.png')
    plt.show()

    to_be_grouped = df[["gender", "race", "age", "readmit_30_days"]].copy()
    to_be_grouped.groupby(['race', 'gender', 'age'], observed=False).readmit_30_days.value_counts().unstack(
        3).plot.barh()
    plt.show()

    '''data_pct_crosstab = pd.crosstab(columns=df['gender'],  # Doesn't work with some NAs
                                    dropna=False,  # See NAs in table
                                    margins=True,  # See row and col totals
                                    normalize='index',
                                    index=df['gender'])'''
    '''ct=pd.crosstab(df.index, [df.race, df.gender])'''
    groups_percentages(df, ['gender', 'race', 'age'])

    X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test = prepare_test_train_datasets(df, random_seed)
    X_train_bal, Y_train_bal, A_train_bal = resample_dataset(X_train, Y_train, A_train)

    if show_train_test:
        figures_test_train(A_train_bal, Y_train_bal, A_test, Y_test)
    '''
    unmitigated_pipeline = Pipeline(steps=[
        ("preprocessing", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000))
    ])

    unmitigated_pipeline.fit(X_train_bal, Y_train_bal)

    Y_pred_proba = unmitigated_pipeline.predict_proba(X_test)[:, 1]
    Y_pred = unmitigated_pipeline.predict(X_test)
    '''

    if use_log_reg:
        Y_pred_proba, Y_pred, unmitigated_pipeline = train_model_lr(X_train_bal, Y_train_bal, X_test)
        display_performance(Y_test, Y_pred_proba, Y_pred)
        if show_coefficients:
            coefficients(unmitigated_pipeline, X_test.columns)
    else:
        Y_pred, unmitigated_pipeline = train_model_hg(X_train_bal, Y_train_bal, X_test)
        display_performance_hg(Y_test, Y_pred)

    #display_performance(Y_test, Y_pred_proba, Y_pred)

    if show_metrics_before:
        metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred, df_test, use_log_reg, False, True)

    to = get_threshold_optimizer(unmitigated_pipeline)
    to.fit(X_train_bal, Y_train_bal, sensitive_features=A_train_bal)
    Y_pred_postprocess = to.predict(X_test, sensitive_features=A_test)

    if show_metrics_after:
        metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred_postprocess, df_test, use_log_reg, True,
                False)

    eg = get_exponential_gradient(unmitigated_pipeline)
    eg.fit(X_train_bal, Y_train_bal, sensitive_features=A_train_bal)
    Y_pred_reductions = eg.predict(X_test, random_state=random_seed)

    if show_metrics_after:
        metrics(metrics_dict, df_test[sensitive_features], Y_test, Y_pred_reductions, df_test, use_log_reg, False, False)

    explore_eg_predictors(eg, X_test, Y_test, A_test)


def coefficients(unmitigated_pipeline, columns):
    coef_series = pd.Series(data=unmitigated_pipeline.named_steps["logistic_regression"].coef_[0], index=columns)
    coef_series.sort_values().plot.barh(figsize=(4, 20), legend=False)
    plt.savefig(generated_dir(True) + 'lr_coef.png')
    plt.show()


def display_performance(Y_test, Y_pred_proba, Y_pred):
    fpr, tpr, threshold = roc_curve(Y_test, Y_pred_proba)
    p = RocCurveDisplay(fpr=fpr, tpr=tpr)
    p.plot()
    plt.show()
    plt.savefig(generated_dir(True) + 'lr_roc_curve.png')
    print(balanced_accuracy_score(Y_test, Y_pred))


def display_performance_hg(Y_test, Y_pred):
    fpr, tpr, threshold = roc_curve(Y_test, Y_pred)
    p = RocCurveDisplay(fpr=fpr, tpr=tpr)
    p.plot()
    plt.show()
    plt.savefig(generated_dir(False) + 'hg_roc_curve.png')
    print(balanced_accuracy_score(Y_test, Y_pred))


def train_model_lr(X_train_bal, Y_train_bal, X_test):
    unmitigated_pipeline = Pipeline(steps=[
        ("preprocessing", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000))
    ])

    unmitigated_pipeline.fit(X_train_bal, Y_train_bal)

    Y_pred_proba = unmitigated_pipeline.predict_proba(X_test)[:, 1]
    Y_pred = unmitigated_pipeline.predict(X_test)
    return Y_pred_proba, Y_pred, unmitigated_pipeline


def train_model_hg(X_train_bal, Y_train_bal, X_test):
    unmitigated_pipeline = Pipeline(steps=[
        ("preprocessing", StandardScaler()),
        ("logistic_regression", HistGradientBoostingClassifier(max_iter=1000))
    ])

    unmitigated_pipeline.fit(X_train_bal, Y_train_bal)

    #Y_pred_proba = unmitigated_pipeline.predict_proba(X_test)[:, 1]
    Y_pred = unmitigated_pipeline.predict(X_test)
    return Y_pred, unmitigated_pipeline


def pairwise_correlation(df):
    corr = pg.pairwise_corr(df, method='pearson')
    corr = corr[["X", "Y", 'r']]
    corr_ = corr[(corr['r'] > .6)]
    print(corr_)


def predictive_validity(df):
    sns.pointplot(y="had_emergency", x="readmit_30_days",
                  data=df, errorbar=('ci', 95), linestyle='none')
    plt.show()

    sns.pointplot(y="had_inpatient_days", x="readmit_30_days",
                  data=df, errorbar=('ci', 95), linestyle='none')
    plt.show()

    sns.catplot(y="had_emergency", x="readmit_30_days", hue="race", data=df,
                kind="point", errorbar=('ci', 95), dodge=True, linestyle='none')

    plt.show()

    sns.catplot(y="had_inpatient_days", x="readmit_30_days", hue="race", data=df,
                kind="point", errorbar=('ci', 95), dodge=True, linestyle='none')

    plt.show()

    sns.catplot(y="had_inpatient_days", x="readmit_30_days", hue="gender", data=df,
                kind="point", errorbar=('ci', 95), dodge=True, linestyle='none')
    plt.show()

    sns.catplot(y="had_inpatient_days", x="readmit_30_days", hue="age", data=df,
                kind="point", errorbar=('ci', 95), dodge=True, linestyle='none')
    plt.show()


def groups_percentages(df, column_names):
    for col_name in column_names:
        count = df[col_name].value_counts()
        percentage = df[col_name].value_counts(normalize=True)
        df_ = pd.concat([count, percentage], axis=1, keys=('Count', 'Percentage'))
        print(col_name + ':', df_)
        df_.to_csv('./generated/' + col_name + '.csv')


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


def pivot(df):
    pv = np.round(pd.pivot_table(df, values='readmit_30_days',
                                 index=['age'],
                                 observed=False,
                                 columns=['gender', 'race'],
                                 aggfunc=np.count_nonzero,
                                 fill_value=0), 0)

    np.round(pd.pivot_table(df, values='readmit_30_days',
                            index=['gender', 'race'],
                            observed=False,
                            columns=['age'],
                            aggfunc=np.count_nonzero,
                            fill_value=0), 0).plot.barh(figsize=(10, 7), title='Mean car price by make and '
                                                                               'number of doors')
    plt.savefig('./generated/' + 'pivot_.png')
    plt.show()

    print(pv)


def prepare_test_train_datasets(df, random_seed):
    target_variable = "readmit_30_days"
    #demographic = ["race", "gender"]
    sensitive = ["race", "gender"]
    Y, A = df.loc[:, target_variable], df.loc[:, sensitive]
    X = pd.get_dummies(df.drop(columns=[
        "race",
        "race_all",
        "discharge_disposition_id",
        "readmitted",
        "readmit_binary",
        "readmit_30_days"
    ]))

    #random_seed = 445
    #np.random.seed(random_seed)

    X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test = train_test_split(
        X,
        Y,
        A,
        df,
        test_size=0.50,
        stratify=Y,
        random_state=random_seed
    )
    return X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test


def resample_dataset(X_train, Y_train, A_train):
    negative_ids = Y_train[Y_train == 0].index
    positive_ids = Y_train[Y_train == 1].index
    balanced_ids = positive_ids.union(np.random.choice(a=negative_ids, size=len(positive_ids)))

    X_train = X_train.loc[balanced_ids, :]
    Y_train = Y_train.loc[balanced_ids]
    A_train = A_train.loc[balanced_ids, :]
    return X_train, Y_train, A_train


def figures_test_train(A_train_bal, Y_train_bal, A_test, Y_test):
    sns.countplot(x="race", data=A_train_bal)
    plt.title("Sensitive Attributes for Training Dataset")
    plt.savefig('./generated/' + 'sa-train.png')
    plt.show()
    sns.countplot(x=Y_train_bal)
    plt.title("Target Label Histogram for Training Dataset")
    plt.savefig('./generated/' + 'tl-histo-train.png')
    plt.show()
    sns.countplot(x="race", data=A_test)
    plt.title("Sensitive Attributes for Testing Dataset")
    plt.savefig('./generated/' + 'sa-test.png')
    plt.show()
    sns.countplot(x=Y_test)
    plt.title("Target Label Histogram for Test Dataset")
    plt.savefig('./generated/' + 'tl-histo-test.png')
    plt.show()


def metrics(metrics_dict, sensitive_features, Y_test, Y_pred, df_test, use_log_reg, use_treshold, unmitigated):
    if unmitigated:
        filename_part = 'unm_'
    elif use_treshold:
        filename_part = 'mit_to_'
    elif not use_treshold:
        filename_part = 'mit_eg_'

    metricframe_unmitigated = MetricFrame(metrics=metrics_dict,
                                          y_true=Y_test,
                                          y_pred=Y_pred,
                                          sensitive_features=sensitive_features)

    # The disaggregated metrics are stored in a pandas Series mf1.by_group:
    print(metricframe_unmitigated.by_group)
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


def get_exponential_gradient(unmitigated_pipeline):
    expgrad_est = ExponentiatedGradient(
        estimator=unmitigated_pipeline,
        constraints=TruePositiveRateParity(difference_bound=0.02)
    )
    return expgrad_est


def explore_eg_predictors(eg, X_test, Y_test, A_test):
    predictors = eg.predictors_
    print(predictors)

    sweep_preds = [clf.predict(X_test) for clf in predictors]
    balanced_error_sweep = [1 - balanced_accuracy_score(Y_test, Y_sweep) for Y_sweep in sweep_preds]
    fnr_diff_sweep = [false_negative_rate(Y_test, Y_sweep, sensitive_features=A_test).difference() for Y_sweep in
                      sweep_preds]
