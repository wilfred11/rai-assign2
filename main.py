import pandas as pd
import seaborn as sns
from sklearn import set_config
from correlation import correlations
from directories import clean_specific_dir, predictive_validity_dir, label_imbalance_dir
from medical import medical, load_dataset, label_imbalance
from predictive_validity import predictive_validity

#pandas.set_option('display.max_columns', None)
#pd.options.mode.copy_on_write = True
pd.set_option("display.float_format", "{:.3f}".format)
set_config(display="diagram")
sns.set()

do = 4
if do == 1:
    medical(show_counts_sf=True, show_pivot=False,
            show_train_test=True, show_coefficients=False, show_metrics_before=True, show_metrics_after=True,
            use_log_reg=False)

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
