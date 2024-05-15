from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.linear_model import LogisticRegression


def get_threshold_optimizer(unmitigated_pipeline):
    postprocess_est = ThresholdOptimizer(
        estimator=unmitigated_pipeline,
        constraints="false_negative_rate_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method='predict_proba'
    )
    return postprocess_est


def get_exponentiated_gradient(unmitigated_pipeline):
    expgrad_est = ExponentiatedGradient(
        estimator=unmitigated_pipeline,
        constraints=TruePositiveRateParity(difference_bound=0.02),
        #sample_weight_name='sample_weight'
        sample_weight_name="logistic_regression__sample_weight",

    )
    return expgrad_est


def get_exponentiated_gradient1(unmitigated_pipeline, random_seed):
    expgrad_est = ExponentiatedGradient(
        estimator=LogisticRegression(max_iter=1000, random_state=random_seed),
        constraints=TruePositiveRateParity(difference_bound=0.02),
        sample_weight_name='sample_weight'

    )
    return expgrad_est
