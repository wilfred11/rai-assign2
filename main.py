from medical import medical


#pd.options.mode.copy_on_write = True

#test()
#model_fairness()
#compas()
#adult_consensus()
medical(show_predictive_validity=True, show_counts_sf=True,  show_pivot=False, show_train_test=True,show_coefficients=False, show_metrics_before=True, show_metrics_after=True, use_log_reg=True)

