from contextlib import contextmanager
import matplotlib as mpl
import seaborn as sns

@contextmanager
def seaborn_theme(style="darkgrid", palette="deep", context="talk"):
    """
    Context manager to temporarily apply a seaborn theme/style,
    then restore matplotlib defaults afterwards.
    """
    old = mpl.rcParams.copy()
    try:
        sns.set_theme(style=style, palette=palette, context=context)
        yield
    finally:
        mpl.rcParams.update(old)