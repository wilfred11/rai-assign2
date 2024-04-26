import pandas as pd

from adult_consensus import adult_consensus
from compas import compas
from medical import medical
from model_fairness import model_fairness
from multiple_features import multiple_features
from test_for_fairness import test

#pd.options.mode.copy_on_write = True

#test()
#model_fairness()
#compas()
#adult_consensus()
medical(show_predictive_validity=False, show_pivot=False, show_train_test=False,show_coefficients=False, show_metrics_before=True, show_metrics_after=True, use_log_reg=False)

