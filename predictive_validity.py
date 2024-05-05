import seaborn as sns
from matplotlib import pyplot as plt

from directories import predictive_validity_dir


def predictive_validity(df, show=True):
    sns.pointplot(y="had_emergency", x="readmit_30_days",
                  data=df, errorbar=('ci', 95), linestyle='none')
    plt.savefig(predictive_validity_dir() + 'pv_readmit_30_had_emergency.png')
    if show:
        plt.show()

    sns.pointplot(y="had_inpatient_days", x="readmit_30_days",
                  data=df, errorbar=('ci', 95), linestyle='none')
    plt.savefig(predictive_validity_dir() + 'pv_readmit_30_had_inpatient_days.png')
    if show:
        plt.show()

    sns.catplot(y="had_emergency", x="readmit_30_days", hue="race", data=df,
                kind="point", errorbar=('ci', 95), dodge=True, linestyle='none')
    plt.savefig(predictive_validity_dir() + 'pv_readmit_30_had_emergency_race.png')
    if show:
        plt.show()

    sns.catplot(y="had_inpatient_days", x="readmit_30_days", hue="race", data=df,
                kind="point", errorbar=('ci', 95), dodge=True, linestyle='none')
    plt.savefig(predictive_validity_dir() + 'pv_readmit_30_had_inpatient_days_race.png')
    if show:
        plt.show()

    sns.catplot(y="had_inpatient_days", x="readmit_30_days", hue="gender", data=df,
                kind="point", errorbar=('ci', 95), dodge=True, linestyle='none')
    plt.savefig(predictive_validity_dir() + 'pv_readmit_30_had_inpatient_days_gender.png')
    if show:
        plt.show()

    sns.catplot(y="had_inpatient_days", x="readmit_30_days", hue="age", data=df,
                kind="point", errorbar=('ci', 95), dodge=True, linestyle='none')
    plt.savefig(predictive_validity_dir() + 'pv_readmit_30_had_inpatient_days_age.png')
    if show:
        plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    sns.countplot(x='race', hue='readmit_30_days', data=df, ax=ax)
    plt.savefig(predictive_validity_dir() + 'pv_readmit_30_race.png')
    if show:
        plt.show()


    plt.clf()
