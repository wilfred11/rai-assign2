import seaborn as sns
from matplotlib import pyplot as plt

from directories import label_imbalance_dir


def label_imbalance(df, show=False):
    print(df["readmit_30_days"].value_counts())  # counts
    print(df["readmit_30_days"].value_counts(normalize=True))  # frequencies

    df["readmit_30_days"].value_counts().plot(kind='barh')
    plt.savefig(label_imbalance_dir() + 'li_readmit_30_days_c.png')
    if show:
        plt.show()

    '''sns.barplot(x="readmit_30_days", data=df, ci=95)
    plt.savefig(label_imbalance_dir() + 'li_readmit_30_days.png')
    if show:
        plt.show()
    '''

    sns.barplot(x="readmit_30_days", y="race", data=df, errorbar=('ci', 95))
    plt.savefig(label_imbalance_dir() + 'li_readmit_race.png')
    if show:
        plt.show()

    sns.pointplot(y="medicaid", x="race", data=df, linestyle='none')
    plt.savefig(label_imbalance_dir() + 'pp_medicaid_race.png')
    if show:
        plt.show()

    to_be_grouped = df[["gender", "race", "age", "readmit_30_days"]].copy()
    to_be_grouped.groupby(['race', 'gender', 'age'], observed=False).readmit_30_days.value_counts().unstack(
        3).plot.barh(figsize=(24, 10))
    plt.savefig(label_imbalance_dir() + 'li_race_gender_age.png')
    if show:
        plt.show()
    plt.clf()
