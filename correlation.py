import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from directories import clean_dirs, correlation_dir, clean_correlation_dirs
from medical import load_dataset, denominalize


def correlations():
    clean_correlation_dirs()
    df = load_dataset()
    df = denominalize(df)
    pairwise_correlation(df, True)
    sensitive_correlation(df)


def pairwise_correlation(df, show=False):
    plt.rcParams["figure.figsize"] = (20, 20)
    print("pairwise correlation")
    sns.heatmap(df.corr().round(2), vmin=-1, vmax=1, annot=True, square=True, cbar=False, annot_kws={'size': 7})
    plt.savefig(correlation_dir() + 'correlations.png', dpi=400)
    if show:
        plt.show()
    plt.clf()


def sensitive_correlation(df):
    cor_df = pd.DataFrame()
    df_ = df.copy()

    filter_col = [col for col in df_ if col.startswith('race')]
    for col in filter_col:
        corr_col_ = df_[df_.columns[0:]].corr()[col]
        new_list = filter_col.copy()
        cor_col = corr_col_.to_frame(col).T.copy()
        cor_col = cor_col.drop(columns=new_list, axis=1)
        cor_df = pd.concat([cor_df, cor_col.copy()], axis=0)
    cor_df.to_csv(correlation_dir() +'corr_race.csv')
