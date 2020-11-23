# Internship

# This project examines EEG data taken from an CLIS-ALS patient. The goal is to investigate potential neurophysiological signatures of consciousness/cognition in these patients which could be used to identify temporal windows for effective BCI communication.

# ANALYSIS PIPELINE
# Use EpochProcessing Class as 'ep'
# first three steps must be done first and in order. Step 4 functions can be done at any time once steps 1-3 completed 
# 1) ep.create_epochs (create epoch data file)
# 2) ep.clean_epochs (clean epoch data file manually)
# 3) ep.divide_epochs (create two files which contain indices of epochs in each brain state)
# 4) ep.erp_analysis (event related potentials) or ep.power_analysis (power spectral analysis) or ep.time_freq_analysis (spectogram analysis)
