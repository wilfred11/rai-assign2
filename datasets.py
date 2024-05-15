import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from directories import generated, test_train_dir
from medical import delete_rows, categorical_features


def prepare_test_train_datasets(df, random_seed):
    print('prepare test train datasets')
    target_variable = "readmit_30_days"
    #demographic = ["race", "gender"]
    sensitive = ["race", "gender"]
    Y, A = df.loc[:, target_variable], df.loc[:, sensitive]
    '''X = pd.get_dummies(df.drop(columns=[
        "race",
        "race_all",
        "discharge_disposition_id",
        "readmitted",
        "readmit_binary",
        "readmit_30_days"
    ]))'''
    X = pd.get_dummies(df.drop(columns=[
        "race",
        "gender",
        "readmit_binary",
        "readmit_30_days"
    ]))
    print('xcols:', X.columns)
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

def figures_test_train(A_train_bal, Y_train_bal, A_test, Y_test, show=False):
    plt.rcParams["figure.figsize"] = (20, 10)
    figures_(A_test, "A_test", show)
    figures_( A_train_bal, "A_train_bal", show)

    sns.countplot(x=Y_train_bal)
    plt.savefig(test_train_dir() + 'y_train_bal_count.png')
    if show:
        plt.show()

    sns.countplot(x=Y_test)
    plt.savefig(test_train_dir() + 'y_test_count.png')
    if show:
        plt.show()
    plt.clf()


def figures_(data, output_name, show=False):
    plt.rcParams["figure.figsize"] = (20, 10)
    data["race"].value_counts().plot(kind='bar', rot=45)
    plt.savefig(test_train_dir() + output_name + '_race_counted.png')
    if show:
        plt.show()

    data["gender"].value_counts().plot(kind='bar', rot=45)
    plt.savefig(test_train_dir() + '_gender_counted.png')
    plt.savefig(test_train_dir() + output_name + '_gender_counted.png')
    if show:
        plt.show()

    dfg = data.groupby(['race', 'gender'], observed=False).size().unstack(level=1)
    dfg.plot(kind='barh')
    plt.savefig(test_train_dir() + output_name + '_grouped_counted.png')
    if show:
        plt.show()
    plt.clf()


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
