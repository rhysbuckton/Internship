# PROJECT GOAL
# This project examines EEG data taken from an CLIS-ALS patient. The goal is to investigate potential neurophysiological signatures of consciousness/cognition in these patients which could be used to identify temporal windows for effective BCI communication.

# ANALYSIS PIPELINE
# Use EpochProcessing Class as 'ep'
# first three steps must be done first and in order. Step 4 functions can be done at any time once steps 1-3 completed 
# 1) ep.create_epochs (create epoch data file)
# 2) ep.clean_epochs (clean epoch data file manually)
# 3) ep.divide_epochs (create two files which contain indices of epochs in each brain state)
# 4) ep.erp_analysis (event related potentials) or ep.power_analysis (power spectral analysis) or ep.time_freq_analysis (spectogram analysis)

# PROJECT ABSTRACT
...
Application of BCI devices for CLIS-ALS patients has so far failed to establish even basic communication. One explanation for this ineffectiveness is that the level of 
cognitive capacity in these patients may vary throughout the day. Previous work in this patient group identified a prominent power spectral peak at sub-alpha frequencies. 
The functional implications of this ostensibly downshifted alpha rhythm are unknown. The first aim of our project was to assess the neurophysiological properties of this 
enigmatic sub-alpha rhythm. To do this, we quantified the amount of event-related desynchronisation (ERD) to trains of auditory stimuli. We found no evidence of any significant 
ERD of the sub-alpha (4 – 6Hz) rhythm. Our second aim was to explore distinct brain states which may be associated with varying levels of cognitive function. We found that our 
patient possessed two distinct brain states associated with a prominent power spectral peak at either sub-alpha (4 – 6Hz) or high-alpha (10.5-12.5Hz) frequencies. Next, we 
explored state-specific responses to auditory stimulation and tested for between-state differences. Our results were inconclusive and so we could not infer whether these brain 
states conferred different levels of cognitive function to our patient. Looking forward, we suggest that an improved and expanded experimental setup may reveal important 
functional distinctions between these two brain states, with important ramifications for the development of improved BCI-based communication strategies. 
...
