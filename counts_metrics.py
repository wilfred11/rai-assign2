import pandas as pd
from fairlearn.metrics import MetricFrame
from matplotlib import pyplot as plt

from directories import generated, unmitigated_dir, mitigated_to_dir, mitigated_eg_dir, clean_specific_dir


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


def groups_percentages(df, column_names):
    for col_name in column_names:
        count = df[col_name].value_counts()
        percentage = df[col_name].value_counts(normalize=True)
        df_ = pd.concat([count, percentage], axis=1, keys=('Count', 'Percentage'))
        print(col_name + ':', df_)
        df_.to_csv(generated() + col_name + '.csv')


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

    metricframe_.by_group.round(2).to_csv(dir + filename_part + "metrics.csv")

    metrics_aggregated = pd.DataFrame({'difference': metricframe_.difference(),
                                       'ratio': metricframe_.ratio(),
                                       'group_min': metricframe_.group_min(),
                                       'group_max': metricframe_.group_max()
                                       }).T
    metrics_aggregated.astype(float).round(2).to_csv(dir + filename_part + "metrics_agg.csv")

    #metrics_.to_excel(generated_dir(use_log_reg) + filename_part + 'sf_metrics.xlsx')
    #metrics_.to_pickle(generated_dir(use_log_reg) + filename_part + 'sf_metrics.pkl')

    metricframe_.by_group.plot.bar(subplots=True, layout=[2, 2], figsize=(12, 12),
                                   legend=False, rot=90, position=.5)
    plt.savefig(dir + filename_part + 'mf.png')
    plt.show()
