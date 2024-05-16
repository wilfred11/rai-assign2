from fairlearn.metrics import selection_rate, false_negative_rate, count
from sklearn.metrics import balanced_accuracy_score


def numeric_and_binary_features():
    return ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses',
            'medicare', 'medicaid', 'had_emergency', 'had_inpatient_days', 'had_outpatient_days', 'readmit_binary',
            'diabetesMed']


def categorical_features():
    return ["race", "gender", "age", "admission_source_id", "medical_specialty", "primary_diagnosis", "max_glu_serum",
            "A1Cresult", "insulin", "change"]


def sensitive_features():
    return ['race', 'gender']


def metrics_dict():
    return {
        "selection_rate": selection_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": balanced_accuracy_score,
        "count": count
    }
