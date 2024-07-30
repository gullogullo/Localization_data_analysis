Code to produce the plots and statistical analysis from the localization test dataset found here: https://zenodo.org/records/5002664

In the code you can modify the flags at lines 322-331 to produce the wanted outputs:
# FLAG FOR PLOTS
want_figures = False
want_qq = False
want_xy_plots = False
# FLAG FOR MIXED MODELS
want_lm = False
# FLAG FOR CORRELATION
want_correlation = False
# FLAG FOR STATS
want_stats = False

You can install the dependencies with pip install -r requirements.txt
Python version used is Python 3.11.0
