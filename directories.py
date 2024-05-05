import os
import shutil


def generated():
    return './generated/'


def generated_dir(use_log_reg):
    if use_log_reg:
        return generated() + 'lr/'
    else:
        return generated() + 'hg/'


def correlation_dir():
    return generated() + 'correlations/'


def clean_dirs():
    try:
        shutil.rmtree(generated())
    except:
        print("An exception occurred")

    try:
        os.mkdir(generated())
    except:
        print("An exception occurred")

    path_lr = os.path.join(generated(), 'lr')
    path_hg = os.path.join(generated(), 'hg')
    try:
        os.mkdir(path_lr)
    except:
        print("An exception occurred")

    try:
        os.mkdir(path_hg)
    except:
        print("An exception occurred")

    try:
        os.mkdir(correlation_dir())
    except:
        print("An exception occurred")


def predictive_validity_dir():
    return generated() + 'predictive_validity/'


def label_imbalance_dir():
    return generated() + 'label_imbalance/'

def sensitive_proxies_dir():
    return generated() + 'sensitive_proxies/'

def clean_correlation_dirs():
    try:
        shutil.rmtree(correlation_dir())
    except:
        print("An exception occurred")

    try:
        os.mkdir(correlation_dir())
    except:
        print("An exception occurred")


def clean_specific_dir(some_dir):
    try:
        shutil.rmtree(some_dir)
    except:
        print("An exception occurred")

    try:
        os.mkdir(some_dir)
    except:
        print("An exception occurred")
