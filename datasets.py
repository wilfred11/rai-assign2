import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from directories import generated, test_train_dir
from settings import categorical_features


def prepare_test_train_datasets(df, random_seed, get_dummies=True):
    print('prepare test train datasets')
    print('random seed:', random_seed)
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

    #print('y index:', )

    if get_dummies:
        X = pd.get_dummies(df.drop(columns=[
            "race",
            "gender",
            "readmit_binary",
            "readmit_30_days"
        ]))
    else:
        X = df.drop(columns=[
            "race",
            "gender",
            "readmit_binary",
            "readmit_30_days"
        ])
    X.replace({False: 0, True: 1}, inplace=True)
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
    print('y_test:', Y_test)
    return X_train, X_test, Y_train, Y_test, A_train, A_test, df_train, df_test

def convert_to_lime_format1(X, categorical_names, col_names=None, invert=False):
    """Converts data with categorical values as string into the right format
    for LIME, with categorical values as integers labels.

    It takes categorical_names, the same dictionary that has to be passed
    to LIME to ensure consistency.

    col_names and invert allow to rebuild the original dataFrame from
    a numpy array in LIME format to be passed to a Pipeline or sklearn
    OneHotEncoder
    """

    # If the data isn't a dataframe, we need to be able to build it
    #print('x.ocls', X.columns)
    #print('len cate names:', len(categorical_names))
    #print('len_col: ',len(X.columns.to_list()))
    if not isinstance(X, pd.DataFrame):
        X_lime = pd.DataFrame(X, columns=col_names)
    else:
        X_lime = X.copy()

    for k, v in categorical_names.items():
        if not invert:
            label_map = {
                str_label: int_label for int_label, str_label in enumerate(v)
            }

        else:
            label_map = {
                int_label: str_label for int_label, str_label in enumerate(v)
            }
        #print("k", k)
        #print("v", v)
        #X_lime.to_csv(generated()+"xlime.csv")
        #print('label_map:', label_map)
        #print("x_lime [:,k]:",X_lime[:,k])
        X_lime[k] = X_lime[k].map(label_map)
    X_lime.to_csv(generated()+"xlime.csv")
    return X_lime



def resample_dataset(X_train, Y_train, A_train):
    negative_ids = Y_train[Y_train == 0].index
    positive_ids = Y_train[Y_train == 1].index
    balanced_ids = positive_ids.union(np.random.choice(a=negative_ids, size=len(positive_ids)))

    X_train = X_train.loc[balanced_ids, :]
    Y_train = Y_train.loc[balanced_ids]
    A_train = A_train.loc[balanced_ids, :]
    print('resample x cols:', X_train.columns)
    #print('resample y cols:', Y_train.columns)
    pd.DataFrame(Y_train, columns=["label"])
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
    print("load dataset")
    data = pd.read_csv(
        "https://raw.githubusercontent.com/fairlearn/talks/main/2021_scipy_tutorial/data/diabetic_preprocessed.csv")

    print(data.head())

    data.to_csv('./data/diabetic.csv', sep=';')

    data = data.drop_duplicates(keep='first')
    print('number of duplicates:', data.duplicated(subset=None, keep='first').sum())

    data = data.drop(columns=[
        "discharge_disposition_id",
        "readmitted",
        #"readmit_30_days"
    ])

    data = delete_rows(data)
    #data["race_all"] = data["race"].copy()
    data["race"] = data["race"].replace({"Asian": "Other", "Hispanic": "Other"})
    data["diabetesMed"] = data["diabetesMed"].replace({"Yes": 1, "No": 0})
    data["A1Cresult"] =  data["A1Cresult"].fillna("NotTested")
    data["max_glu_serum"] =  data["max_glu_serum"].fillna("NotTested")

    # Show the values of all binary and categorical features
    categorical_values = {}
    '''for col in data:
        if col not in {'time_in_hospital', 'num_lab_procedures',
                       'num_procedures', 'num_medications', 'number_diagnoses'}:
            categorical_values[col] = pd.Series(data[col].value_counts().index.values)'''
    #categorical_values_df = pd.DataFrame(categorical_values).fillna('')
    #categorical_values_df.T

    for col_name in categorical_features():
        data[col_name] = data[col_name].astype("category")
    return data


def delete_rows(df):
    df.drop(df[df['gender'] == 'Unknown/Invalid'].index, inplace=True)
    print('gender unique:', df.gender.unique())
    return df
