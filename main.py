import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import set_config

from correlation import correlations
from datasets import load_dataset
from directories import clean_specific_dir, predictive_validity_dir, label_imbalance_dir, sensitive_proxies_dir
from label_imbalance import label_imbalance
from medical import medical, shap, lime2
from predictive_validity import predictive_validity
from proxies_sensitive_features import proxies_for_sensitive_features

# pandas.set_option('display.max_columns', None)
# pd.options.mode.copy_on_write = True
pd.set_option("display.float_format", "{:.3f}".format)
set_config(display="diagram")
sns.set()

# initjs()

random_seed = 422
np.random.seed(random_seed)

do = 1
if do == 1:
    medical(show_counts_sf=True, show_pivot=False,
            show_train_test=True, show_coefficients=True, show_metrics_before=True, show_metrics_after=True,
            use_log_reg=True)

if do == 2:
    correlations()

if do == 3:
    df = load_dataset()
    clean_specific_dir(predictive_validity_dir())
    predictive_validity(df, show=True)

if do == 4:
    df = load_dataset()
    clean_specific_dir(label_imbalance_dir())
    label_imbalance(df, show=True)

if do == 5:
    df = load_dataset()
    clean_specific_dir(sensitive_proxies_dir())
    proxies_for_sensitive_features(df, show=True)

if do == 6:
    shap(random_seed)

if do == 7:
    lime2(random_seed)
