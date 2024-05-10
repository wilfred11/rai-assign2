import seaborn as sns
from matplotlib import pyplot as plt
from directories import sensitive_proxies_dir


def proxies_for_sensitive_features(df, show=False):
    sns.pointplot(y="medicaid", x="race", data=df, linestyle='none')
    plt.savefig(sensitive_proxies_dir() + 'psf_medicaid_race.png')
    if show:
        plt.show()
