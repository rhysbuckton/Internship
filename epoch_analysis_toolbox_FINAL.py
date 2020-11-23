#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#------------------------------------------------------------------------
# Requirements: 
#       ensure data files, mbp module and this module  in cd
#       mne eeg toolbox
#       designed for use with jupyter notebook
# Author: Rhys Buckton
#------------------------------------------------------------------------



class EpochProcessing:
    
    '''
    Stores modules for analysis of epochs from EEG data
    '''
    
    
#--------------------------------------------------------------------------------------
# IDENTIFY STIMULUS TIMEPOINTS AND CONSTRUCT EPOCHS
#----------------------------------------------------------------------------------------

    def create_epochs(file_list = ['patient_gr_wuerzburg_29_11_19.edf', 'patient_gr_wuerzburg_30_11_19.edf', 
                               'patient_gr_wuerzburg_12_12_19.edf', 'patient_gr_wuerzburg_13_12_19.edf', 
                               'patient_gr_wuerzburg_17_12_19.edf', 'patient_gr_wuerzburg_18_12_19.edf', 
                               'patient_gr_wuerzburg_21_01_20.edf', 'patient_gr_wuerzburg_22_01_20.edf', 
                               'patient_gr_wuerzburg_29_01_20.edf', 'patient_gr_wuerzburg_30_01_20.edf'], 
                  eeg_chans = ['CH1 F7-O1', 'CH3 Fp1-F8', 'CH4 F8-F7', 'CH5 Fp1-O1', 'CH7 FP1-F7'], stim_chan = ['RED'],
                     output_file = 'all_epochs_GW-epo.fif'):

        '''
        This function iterates through each given .edf file, defining potential stimuli events and using these events to 
        construct mne epoch files. These epoch files are combined across sessions into a single file. An interactive
        window allows for subsequent manual rejection of 'false' epochs and the epoch (-epo.fif) file is saved to cd.

        INPUTS:
            file_list: an array of strings indicating a names of files which contain the recording data for each session
            eeg_chans: array of strings indicating good eeg chans to use for analysis
            stim_chan: single element containing string of channel used for identifying stimuli (RED or INFRARED)
            output_file: give a string name to the outputed epoch file 

        OUTPUTS:
            output_file: saves a single file containing the combined epoch data to current directory
            graphs displaying times of stimuli events across each session
        '''
        
        #Import modules
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import copy
        import mne                          # available eeg processing toolbox
        import sys
        from itertools import compress


        all_chans = np.append(eeg_chans, stim_chan)  # combine eeg and stim chans

        # Identify potential stimulus timepoints for each file
        for curfile in file_list:
            # load data
            file = curfile
            print('Extracting potential stimulus timepoints from: ' + np.str(file))
            raw = mne.io.read_raw_edf( file, preload=True)
            #select good eeg chans and stim chan
            raw.pick_channels(ch_names = all_chans)
            # process data
            info = raw.info
            fs = info['sfreq']
            z, t = raw[0,:] # EEG
            stim_index = np.where(all_chans == stim_chan)[0]
            x, t = raw[stim_index,:] # RED channel as stim channel
            # extract 50Hz signal
            x_ind = range(0,z.shape[1],5) 
            x = x[:,x_ind]

            # create boolean array for timepoints where value crosses above the threshold value
            stim_data = x[0] # stim channel
            threshold = np.std(stim_data)    # set threshold based on standard deviation
            peaklogic = np.empty(len(stim_data), dtype=bool)
            peaklogic[:] = np.nan
            for i in range(len(stim_data)):
                if i == 0:
                    peaklogic[0] = 0 # ignore first value
                elif stim_data[i] > threshold and stim_data[i-1] <= threshold:
                    peaklogic[i] = 1 # indicates points where stim channel crosses the threshold value
                else: 
                    peaklogic[i] = 0 

            # re-adjust threshold value and re-do if number of potential is too high
            while np.sum(peaklogic) > 500:
                print('increasing threshold value')
                threshold = threshold * 1.1
                peaklogic = np.empty(len(stim_data), dtype=bool)
                peaklogic[:] = np.nan
                for i in range(len(stim_data)):
                    if i == 0:
                        peaklogic[0] = 0 # ignore first value
                    elif stim_data[i] > threshold and stim_data[i-1] <= threshold:
                        peaklogic[i] = 1 # indicates points where stim channel crosses the threshold value
                    else: 
                        peaklogic[i] = 0 


            #create array containing the indices of timepoints of potenital stimuli
            peaktimes = list(compress(range(len(peaklogic)), peaklogic))
            print('number of potenital events, before cleaning is:' ,len(peaktimes))

            # clean data by removing values which are too close. This will save time when visually inspecting epochs
            cleanwidth = -(0.1 * 50) # number of timestamps in 0.1 seconds = 0.1 * sample freq 
            new_peaktimes = np.copy(peaktimes)
            for i in range (len(peaktimes)):
                if i == 0:
                    pass
                elif peaktimes[i-1]-peaktimes[i]<cleanwidth:
                    pass
                elif peaktimes[i-1]-peaktimes[i]>=cleanwidth:
                    new_peaktimes[i] = new_peaktimes[i-1]
            new_peaktimes = np.unique(new_peaktimes) # remove duplicate stimulus times
            print('number of potential events, after cleaning, is:' ,len(new_peaktimes))

            # re-adjust to turn timepoints from a 50Hz array to 250Hz array 
            for i in range(len(new_peaktimes)):
                new_peaktimes[i] = new_peaktimes[i] * 5 #adjust sampling freq

            # create array suitable for mne toolbox
            new_peaktimes = np.int_(new_peaktimes)
            secondcol = np.zeros(len(new_peaktimes),dtype='int32')   #Length of event (zero in this case)
            thirdcol = np.ones(len(new_peaktimes),dtype='int32')     #Label of event type (all the same in this case)
            peakevents = np.stack((new_peaktimes,secondcol,thirdcol),axis=-1)

            # display times of potential stimuli
            fig = plt.rcParams["figure.figsize"]= [12, 1]
            fig = mne.viz.plot_events(peakevents, sfreq= 250, first_samp = raw.first_samp)

            # construct epochs based on potential stim times
            print('constructing epochs for new file')
            epochs = mne.Epochs(raw, peakevents, tmin=-16, tmax=16, preload = True) #base:-16 to 0s, resp:0 to 16s
            del raw #delete raw data to free memory

            print('adding epochs from new file')
            if file == file_list[0]:
                all_epochs = epochs
            else:
                epochs_list = [all_epochs, epochs]
                all_epochs = mne.concatenate_epochs(epochs_list, add_offset = True)   

        # interactive plot for rejecting false-epochs
        print('INSTRUCTION: MANUALLY EXCLUDE EPOCHS WHICH ARE NOT CENTRED AROUND THE ONSET OF TRUE STIMULUS TRAINS')
        #get_ipython().run_line_magic('matplotlib', 'notebook')
        matplotlib.use('TkAgg')
        mne.Epochs.plot(all_epochs, picks = stim_chan, n_epochs = 1, n_channels = 1, show = True, block = True)        
    
        # save output epoch file to cd
        mne.Epochs.save(all_epochs, output_file)




#----------------------------------------------------------------
#CLEAN EPOCHS 
#----------------------------------------------------------------

    def clean_epochs(input_file = 'all_epochs_GW-epo.fif', output_file = 'clean_epochs_GW-epo.fif', 
                      eeg_chans = ['CH1 F7-O1', 'CH3 Fp1-F8', 'CH4 F8-F7', 'CH5 Fp1-O1', 'CH7 FP1-F7']):
        
        '''
        This function takes the epoch file created by create_epochs and provides an interactive window for manual
        rejection of epochs containing obvious artefacts or signal-saturation events. Based on an apparent sychronisation
        error between stim and eeg chans an offset can be added to realign these channels. The cleaned epoch file is saved
        to the cd.

        INPUT:
            input_file: string corresponding to name of output file from create_epochs function
            output_file: set desired string name of output of current function
            eeg_chans: array containing good EEG chan names in string format

        OUPUT:
            output_file: cleaned epoch file is saved to cd, name defined as input
        '''

        #import modules
        import numpy as np
        import mne
        import matplotlib
        
        #load data
        all_epochs = mne.read_epochs(input_file, preload= True)

        # add offset
        #offset = 0.5
        #all_epochs.shift_time(tshift = offset, relative=True) # compensate for 0.5d drift between pleth and eeg signal
        #print('Compensate for eeg-stim channel temporal discrepancy by adding offset of  '+ np.str(offset) + ' seconds')
        
        # interactive plot for rejecting bad epochs
        print('INSTRUCTION: MANUALLY REJECT EPOCHS CONTAINING SATURATION POINTS OR OTHER ARTEFACTS')
        all_epochs.baseline = (-15, 0)
        all_epochs.pick_channels(ch_names = eeg_chans)
        matplotlib.use('TkAgg')
        mne.Epochs.plot(all_epochs, n_epochs = 1, n_channels = 1, block = True)

        #save epochs file to current directory
        mne.Epochs.save(all_epochs, output_file)
        



#------------------------------------------------------------------------------
# DIVIDE EPOCHS INTO HIGH AND LOW BRAIN STATES
#----------------------------------------------------------------------------

    def divide_epochs(input_file = 'all_epochs2_GW-epo.fif', channel = 'CH5 Fp1-O1', freq_lim = [4.5, 6], thresh = -6.55):
    
        '''
        This function uses fft to generate resting state power spectra for all epochs. From this data, averaged power 
        values for a given channel and freq range are used to divide epochs into 2 putative brain states (high and low).
        The number of epochs in each state are equalised by random subselection from the state with more epochs. 
        Two arrays are saved to the cd, containing the epoch indices corresponding to each brain state.

        INPUT:
            input_file: name of file with stored epoch data - the output of clean_epochs function
            channel: channel used for the division of epochs, string format
            freq_lim: numerical array with two values - low limit followed by high limit: [x, y]
            thresh: threshold average power value used to split brain states

        OUTPUT:
            high_epochs: array containing indices of high state epochs
            low_epochs: array containing indices of low state epochs
            histograms of average power values of the brain rhythm of interest
        '''

        #Import modules
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import copy
        import mne                          # available eeg processing toolbox
        import mbp                          # dft processing module from Moritz
        import sys

        all_epochs = mne.read_epochs(input_file, preload= True)
        chan_index = all_epochs.ch_names.index(channel)

        # Generate np array with only baseline segments
        base_epochs = mne.Epochs.copy(all_epochs)
        base_epochs = mne.Epochs.crop(base_epochs, tmin = -15, tmax = 0)
        base_epochs = mne.Epochs.get_data(base_epochs)

        # initialise power analysis
        start_freq = 0.5
        end_freq = 15.0001
        res = 0.1
        dft_freqs = np.arange(start_freq, end_freq, res) # select frequencies of interest
        base_dft = np.zeros((base_epochs.shape[1], base_epochs.shape[0], len(dft_freqs)-1)) # create empty dft array                    (nchans, nepochs, nfreq)

        #iterate through channels and calculate dft
        for i in range(base_epochs.shape[1]):
            base_dft[i,:,:] = np.squeeze(mbp.mbp(base_epochs[:,i,:], freqs = dft_freqs, fs = 250)) 

        # create array indicating frequencies within the band used for division of brain states
        adj_freqs = np.linspace(start_freq + (res/2), end_freq - (res/2), num = len(dft_freqs)-1)
        freq_index = np.where((adj_freqs >= freq_lim[0]) & (adj_freqs <= freq_lim[1]))[0] 
        print('N.o freqs within band of interest ', len(freq_index))

        #iterate through epochs and calculate average power in 4.5 to 6 Hz range
        avg_bandpower = np.zeros( base_dft.shape[1]) # create empty array ( n_epochs)
        for j in range(base_dft.shape[1]):
                avg_bandpower[j] = np.mean(base_dft[chan_index,j,freq_index], axis = 0) 

        # Plot histograms of avg 4.5-6Hz power across individual epochs 
        plt.figure( figsize = (10,2))
        plt.hist(avg_bandpower, bins = 30)
        plt.title(np.str(channel) + ': Histogram of Avg Bandpower in range ' + np.str(freq_lim) + ' (Hz)')
        plt.xlabel('Log Bandpower')
        plt.ylabel('Frequency')
        plt.axvline(x= thresh, color = 'r')
        plt.show()

        #divide epochs based on whether their avg power value is above or below set threshold 
        high_index = np.where(avg_bandpower > thresh)[0] # indices of high brain state epochs
        print('The number of high brain state epochs:', len(high_index))
        low_index = np.where(avg_bandpower <= thresh)[0] # indices of low brain state epochs
        print('The number of low brain state epochs:', len(low_index))

        # Equalise number of low vs high brain state epochs used to avoid bias
        print('Equalising number of epochs per condition to avoid bias')
        high_count = len(high_index)
        low_count = len(low_index)
        if high_count > low_count: #if high state has more, select random subset from high epochs
            adj_high = np.random.choice(high_index, size = low_count, replace = False)
            adj_low = low_index
        elif high_count < low_count: # if low state has more, select random subset from low epochs
            adj_low = np.random.choice(low_index, size = high_count, replace = False)
            adj_high = high_index
        else: 
            pass 
        print('Same number of epochs in each condition? ', len(adj_low) == len(adj_high)) # sanity check
        print('Adjusted number of epochs for both conditions: ', np.str(len(adj_low)))    

        # save files to cd
        np.save('high_epochs.npy', adj_high)
        np.save('low_epochs.npy', adj_low)
        print('saving high and low epoch index arrays to cd')

        
  

#------------------------------------------------------------------------------
# ANALYSIS OF EVENT RELATED POTENTIALS 
#-------------------------------------------------------------------------------
        
    def erp_analysis(input_file = 'all_epochs3_GW-epo.fif', high_file = 'high_epochs.npy', low_file = 'low_epochs.npy',
                       perm_num = 1000):
    
        '''
        This function performs ERP of analysis of the combined-epoch data and also compares ERPs of each brain state.
        Statistics applied to test for sig differences between the ERPs of each brain state at each response timepoint.

        INPUT:
            input_file: file containing epoch data - the output of epochs_creation
            high_file: name of file containing indicies of high-state epochs
            low_file: name of file containing indicies of low-state epochs
            perm_num: num of permutations for non-paramaetric stats

        OUTPUT:
            graphs of combined ERPs for each EEG channel
            graph including high and low state ERPs alongside graphs displaying the difference (stats shown)
        '''
    
        #Import modules
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import copy
        import mne                          # available eeg processing toolbox
        import sys
        from random import randrange
        from itertools import compress

        #load, filter and downsample data
        high_epochs = np.load(high_file)
        low_epochs = np.load(low_file)
        filt_epochs = mne.read_epochs(input_file, preload= True)
        mne.Epochs.filter(filt_epochs, l_freq = 0.1, h_freq = 15) # low pass filter, 15Hz cut off
        filt_epochs.resample(50, npad='auto') # downsample
        filt_epochs.baseline = None
        
        # split 15s epochs into 3s epochs
        print('splitting 15s epochs into 3s epochs')
        short_indices = np.where(np.logical_and(filt_epochs.times >=0, filt_epochs.times<3))
        short_times = filt_epochs.times[short_indices]
        resp_1 = filt_epochs.copy().crop(tmin = 0, tmax = 3, include_tmax = False)
        resp_2 = filt_epochs.copy().crop(tmin = 3, tmax = 6, include_tmax = False)
        resp_2.shift_time(tshift = 0, relative = False)
        resp_3 = filt_epochs.copy().crop(tmin = 6, tmax = 9, include_tmax = False)
        resp_3.shift_time(tshift = 0, relative = False)
        resp_4 = filt_epochs.copy().crop(tmin = 9, tmax = 12, include_tmax = False)
        resp_4.shift_time(tshift = 0, relative = False)
        resp_5 = filt_epochs.copy().crop(tmin = 12, tmax = 15, include_tmax = False)
        resp_5.shift_time(tshift = 0, relative = False)
        epoch_list = [resp_1, resp_2, resp_3, resp_4, resp_5]
        short_epochs = mne.concatenate_epochs(epoch_list)

        #plot ERPs for all channels individually
        #for i in range(len(short_epochs.ch_names)):
        #    chan = short_epochs.ch_names[i]
        #    mne.Epochs.plot_image(short_epochs, picks = chan, combine = None, title = np.str(chan) + ': ERP', vmin = -100,
        #                         vmax = 100)

        #input high and low index arrays and re-divide epochs
        #divide low state epochs
        low_filt = filt_epochs[low_epochs]
        resp_1 = low_filt.copy().crop(tmin = 0, tmax = 3, include_tmax = False)
        resp_2 = low_filt.copy().crop(tmin = 3, tmax = 6, include_tmax = False)
        resp_2.shift_time(tshift = 0, relative = False)
        resp_3 = low_filt.copy().crop(tmin = 6, tmax = 9, include_tmax = False)
        resp_3.shift_time(tshift = 0, relative = False)
        resp_4 = low_filt.copy().crop(tmin = 9, tmax = 12, include_tmax = False)
        resp_4.shift_time(tshift = 0, relative = False)
        resp_5 = low_filt.copy().crop(tmin = 12, tmax = 15, include_tmax = False)
        resp_5.shift_time(tshift = 0, relative = False)
        epoch_list = [resp_1, resp_2, resp_3, resp_4, resp_5]
        short_low = mne.concatenate_epochs(epoch_list)

        #divide high state epochs
        high_filt = filt_epochs[high_epochs]
        resp_1 = high_filt.copy().crop(tmin = 0, tmax = 3, include_tmax = False)
        resp_2 = high_filt.copy().crop(tmin = 3, tmax = 6, include_tmax = False)
        resp_2.shift_time(tshift = 0, relative = False)
        resp_3 = high_filt.copy().crop(tmin = 6, tmax = 9, include_tmax = False)
        resp_3.shift_time(tshift = 0, relative = False)
        resp_4 = high_filt.copy().crop(tmin = 9, tmax = 12, include_tmax = False)
        resp_4.shift_time(tshift = 0, relative = False)
        resp_5 = high_filt.copy().crop(tmin = 12, tmax = 15, include_tmax = False)
        resp_5.shift_time(tshift = 0, relative = False)
        epoch_list = [resp_1, resp_2, resp_3, resp_4, resp_5]
        short_high = mne.concatenate_epochs(epoch_list)

        # extract avg ERPs for the two brain states
        print('generating avg ERPs for each brain state')
        low_data = short_low.get_data()  
        low_erp = np.mean(low_data, axis = 0) # avg low state epochs
        low_error = np.std(low_data, axis = 0)
        high_data = short_high.get_data()
        high_erp = np.mean(high_data, axis = 0)
        high_error = np.std(high_data, axis = 0)

        # compute permutation statistics for each timepoint
        num_chans = high_data.shape[1]
        num_times = high_data.shape[2]
        print('computing permutation stats')
        #perm_num = 1000
        print('number of permutations: ', perm_num)
        print('number of time points tested: ', num_times)
        perm_stat = np.zeros([num_chans, num_times])
        for c in range(num_chans):
            print('generating stats for ', np.str(filt_epochs.ch_names[c]))
            for i in range(num_times):
                avg_low = np.mean(low_data[:, c, i])
                avg_high = np.mean(high_data[:, c, i])
                real_diff = np.absolute(avg_low - avg_high)

                #permutate by randomising between high and low epochs
                shuffle_data = np.concatenate([low_data[:, c, i], high_data[:, c, i]])
                count = np.zeros([perm_num])                            
                shuffle_diff = np.zeros([perm_num])
                for k in range(perm_num):
                    np.random.shuffle(shuffle_data)
                    shuffle_low, shuffle_high = np.split(shuffle_data, 2)
                    shuffle_diff[k] = np.absolute(np.mean(shuffle_low) - np.mean(shuffle_high))
                    if shuffle_diff[k] >= real_diff:
                        count[k] = 1
                    else: 
                        count[k] = 0    
                perm_stat[c, i] = np.sum(count)/perm_num 

        #benjamin hochberg correction for multiple comparisons
        print('computing Benjamin Hochberg correction for multiple comparisons')
        rank_array = np.zeros([num_chans, num_times])
        for c in range(num_chans):
            rank_array[c, :] = perm_stat[c, :].copy().argsort().argsort()
        rank_array = rank_array + 1
        fdr = 0.25 #false discovery rate as a decimal
        print('number of comparisons/tests adjusted for: ', num_times)
        c_value = rank_array * (fdr/num_times)
        sig_array = np.zeros([num_chans, num_times])

        # create array indicating whether p values at each timepoint exceed c value
        for i in range(num_times): 
            if perm_stat[c, i] <= c_value[c, i]:
                sig_array[c, i] = 1
            else: 
                sig_array[c, i] = 0

        print('number of significant time points across...')
        print('all channels: ', sig_array.sum())
        for c in range(num_chans):
            print(np.str(filt_epochs.ch_names[c]), ': ', np.sum(sig_array[c, :]))

        # plot graphs of high vs low erps and difference between brain states with significance indicated by asterisks
        time_axis = short_low.times
        diff_data = low_data - high_data
        labels = ['Low state', 'High state']
        font = {'size'   : 12}
        plt.rc('font', **font)
        fig = plt.figure( figsize = (14, 14))
        for c in range(num_chans):
            #plot average erps
            plt.subplot(num_chans, 2, (c*2)+1)
            line1 = plt.plot(time_axis, high_erp[c,:], color = 'r', label = labels[1])
            plt.fill_between(time_axis, high_erp[c,:] - high_error[c,:], high_erp[c,:] + high_error[c,:], facecolor = 'r',                              alpha=0.2)
            line2 = plt.plot(time_axis, low_erp[c,:], color = 'b', label = labels[0])
            plt.fill_between(time_axis, low_erp[c,:] - low_error[c,:], low_erp[c,:] + low_error[c,:], facecolor = 'b',                              alpha = 0.2)
            plt.title(str(filt_epochs.ch_names[c]) + ': ERPs')
            plt.axvline(x= 0, color = 'r')
            plt.legend(loc = 'upper right', prop={'size': 10})
            if c == num_chans-1:
                plt.xlabel('Time (s)')
                plt.ylabel('Potential')
            #plot diff between high and low with stats included
            plt.subplot(num_chans, 2, (c*2)+2)
            average = np.mean(diff_data[:, c, :], axis = 0)
            error = np.std(diff_data[:, c, :], axis = 0)
            plt.plot(time_axis, average, color = 'g')
            plt.fill_between(time_axis, average - error, average + error, facecolor = 'g', alpha=0.2)
            plt.title(str(filt_epochs.ch_names[c]) + ': Low - High State')
            plt.axvline(x= 0, color = 'r')
            sig_times = time_axis[np.nonzero(sig_array[c, :])]
            max_y = np.amax(average)
            y_constant = np.zeros([len(sig_times)]) + max_y + (max_y*0.2)
            plt.plot(sig_times, y_constant, color = 'k', marker = "*", linestyle = ":")
        plt.legend([line1, line2], ['high', 'low'], loc = 'upper right', prop={'size': 12})
        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(wspace=0.25)
        plt.show()
        
        
        
        
        
#-----------------------------------------------------------------------
# POWER SPECTRAL ANALYSIS
#-------------------------------------------------------------------------


    def power_analysis(input_file = 'all_epochs2_GW-epo.fif', start_freq = 0.5, end_freq = 15.0001, freq_res = 0.25, 
                  alpha_lim = [4, 6], high_file = 'high_epochs.npy', low_file = 'low_epochs.npy', perm_num = 1000):

        
        '''
        This function applies fft to calculate power spectra for discrete time windows (time dimension is reduced). 
        It plots baseline vs response spectra and then evaluates sig differences in a restricted freq range. Power
        spectral data is subvided into high and low state epochs and permutation-based analysis tested for sig diff
        between brain states and also between baseline and response segments. 

        INPUT:
            input_file: file containing epoch data - output of clean_epochs
            start_freq: min freq for power spectra
            end_freq: max freq for power spectra
            freq_res: frequency resolution
            alpha_lim: array containing lower and then higher frequency limit for focused analysis of ERD
            high_file: saved array with indices of high-state epochs - output of divide_epochs
            low_file: saved array with indices of low-state epochs - output of divide_epochs
            perm_num: number of permutations for non-parametric stats
        
        OUTPUT:
            graph displaying power spectra of baseline and response segments for all epochs
            plot and test between baseline and response in specific, narrow freq band
            graph showing and testing for differences in baseline vs response (separately for high and low epochs)
            graph showing and testing for differences in high vs low state (separately for BASE and RESP segments)
        '''
        
        #Import modules
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import copy
        import mne                          # available eeg processing toolbox
        import mbp                          # dft processing module from Moritz
        import sys
        from random import randrange
        from itertools import compress


        # GENERATE POWER SPECTRA: COMPARE RESTING VS BASELINE

        #load data
        high_epochs = np.load(high_file)
        low_epochs = np.load(low_file)
        all_epochs = mne.read_epochs(input_file, preload= True)

        # power analysis of baseline
        base_epochs = mne.Epochs.copy(all_epochs)
        base_epochs = mne.Epochs.crop(base_epochs, tmin = -15, tmax = 0)
        base_epochs = mne.Epochs.get_data(base_epochs)
        dft_freqs = np.arange(start_freq, end_freq, freq_res) # select frequencies of interest
        num_epochs = base_epochs.shape[0]
        num_chans = base_epochs.shape[1]
        num_freqs = len(dft_freqs) - 1
        base_dft = np.zeros((num_chans, num_epochs, num_freqs)) # empty dft array of (nchans, nepochs, nfreq)
        #iterate through channels and calculate dft
        for i in range(num_chans):
            base_dft[i,:,:] = np.squeeze(mbp.mbp(base_epochs[:,i,:], freqs = dft_freqs, fs = 250)) 

        # power analysis of response segment
        resp_epochs = mne.Epochs.copy(all_epochs)
        resp_epochs = mne.Epochs.crop(resp_epochs, tmin = 0, tmax = 15) # response period 14s to 29s after stimulus train
        resp_epochs = mne.Epochs.get_data(resp_epochs)
        num_epochs = resp_epochs.shape[0]
        resp_dft = np.zeros((num_chans, num_epochs, num_freqs)) # dft array (nchans, nepochs, nfreq)
        #iterate through channels and calculate dft
        for i in range(num_chans):
            resp_dft[i,:,:] = np.squeeze(mbp.mbp(resp_epochs[:,i,:], freqs = dft_freqs, fs = 250)) 

        # check expected dimensions
        print('Number of channels: ', num_chans)
        print('Number of epochs: ', num_epochs)
        print('Number of output frequencies:', num_freqs)

        #plot baseline vs response power spectra
        font = {'size'   : 20}
        ps_labels = ['Baseline', 'Response']
        adj_freqs = np.linspace(start_freq + (freq_res/2), end_freq - (freq_res/2), num = num_freqs) #  
        fig = plt.figure( figsize = (35,15))
        plt.rc('font', **font)
        for i in range(base_dft.shape[0]):
            plt.subplot(2,3,i+1)
            avg_base = np.mean(base_dft[i,:,:], axis = 0)
            base_error = np.std(base_dft[i,:,:], axis = 0)
            avg_resp = np.mean(resp_dft[i,:,:], axis = 0)
            resp_error = np.std(resp_dft[i,:,:], axis = 0)
            plt.plot(adj_freqs, avg_base, color = 'b', label = ps_labels[0])
            plt.fill_between(adj_freqs, avg_base - base_error, avg_base + base_error, facecolor = 'b', alpha=0.2)
            plt.plot(adj_freqs, avg_resp, color = 'r', label = ps_labels[1])
            plt.fill_between(adj_freqs, avg_resp - resp_error, avg_resp + resp_error, facecolor = 'r', alpha=0.2)
            plt.title(all_epochs.ch_names[i])
            plt.legend(loc = 'upper right')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Log Power')
        fig.subplots_adjust(hspace=0.3)
        plt.show()


        # STATISTICAL EVALUATION OF DIFFERENCES BETWEEN BASELINE AND RESPONSE IN SUBSELECTED FREQ RANGE AND CHANNEL
        # for each individual frequency within 4.5 to 6Hz. Generate permutation distribution of test statistic: avg                 #baseline 
        # power - avg response power. One sided test of significant difference i.e. is alpha power lower in response 

        # subselect data (freqs within range) 
        alpha_freqs = np.where((adj_freqs >= alpha_lim[0]) & (adj_freqs <= alpha_lim[1]))[0]
        freq_labels = adj_freqs[alpha_freqs]
        alpha_base = base_dft[:, :, alpha_freqs]
        alpha_resp = resp_dft[:, :, alpha_freqs]

        # generate permutation stats
        print('computing permutation stats with Bonferroni correction')
        #perm_num = 1000
        perm_stat = np.zeros((num_chans, len(alpha_freqs)))
        count = np.zeros([perm_num])
        for c in range(num_chans):        
            for i in range(len(alpha_freqs)): # iterate across frequencies of interest
                base_data = alpha_base[c, :, i]
                resp_data = alpha_resp[c, :, i]
                real_diff = np.mean(base_data) - np.mean(resp_data)

                shuffle_data = np.concatenate([base_data, resp_data], axis = 0)
                sample_diff = np.zeros([perm_num])
                for j in range(perm_num):
                    np.random.shuffle(shuffle_data)
                    sample_base, sample_resp = np.split(shuffle_data, 2)
                    sample_diff[j] = np.mean(sample_base) - np.mean(sample_resp)
                    if sample_diff[j] >= real_diff:
                        count[j] = 1
                    else: 
                        count[j] = 0    
            
                perm_stat[c, i] = len(alpha_freqs) * (np.sum(count)/perm_num) #bonferroni corrected

        # generate boolean array indicating significant differences between base and resp
        sig_array = np.zeros((num_chans, len(alpha_freqs)))
        for c in range(num_chans):
            for i in range(len(alpha_freqs)):
                if perm_stat[c, i] < 0.05:
                    sig_array[c, i] = 1
                else:
                    sig_array[c, i] = 0

        # plot power spectra of base and resp segments in specific freq band, indicate significant differences
        ps_labels = ['Baseline', 'Response']
        font = {'size'   : 11}
        plt.rc('font', **font)
        fig = plt.figure( figsize = (15,6))
        for c in range(num_chans):
            plt.subplot(2, 3, c+1)
            avg_base = np.mean(alpha_base[c, :, :], axis = 0)
            avg_resp = np.mean(alpha_resp[c, :, :], axis = 0)
            error_base = np.std(alpha_base[c, :, :], axis = 0)
            error_resp = np.std(alpha_resp[c, :, :], axis = 0)
            #plot averages and standard deviation
            plt.plot(freq_labels, avg_base, label = ps_labels[0], color = 'b') # plot baseline average
            plt.fill_between(freq_labels, avg_base - error_base, avg_base + error_base, facecolor = 'b', alpha=0.2) # std
            plt.plot(freq_labels, avg_resp, label = ps_labels[1], color = 'r') # plot resp average
            plt.fill_between(freq_labels, avg_resp - error_resp, avg_resp + error_resp, facecolor = 'r', alpha=0.2) # std
            #sig stars
            sig_freqs = alpha_freqs[np.nonzero(sig_array[c, :])]    
            y_constant = np.zeros([len(sig_freqs)]) - 5
            plt.plot(sig_freqs, y_constant, color = 'k', marker = "*", linestyle = ":")
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Log Power')
            plt.legend(loc = 'upper right')
            plt.title(all_epochs.ch_names[c] + ': Power Spectra')
            fig.subplots_adjust(hspace=0.5)
            fig.subplots_adjust(wspace=0.3)
        plt.show()




        #TEST FOR DIFFERENCES IN BASELINE VS RESPONSE SEGMENTS (separately for each brain state)

        num_chans = base_dft.shape[0]
        num_freqs = base_dft.shape[2]

        low_stats = np.zeros([num_chans, num_freqs])
        high_stats = np.zeros([num_chans, num_freqs])
        #perm_num = 1000
        print('number of permutations: ', perm_num)

        for c in range(num_chans):
            print('generating stats for ', np.str(all_epochs.ch_names[c]))
            for i in range(num_freqs): 
                base_low = np.mean(base_dft[c, low_epochs, i])
                base_high = np.mean(base_dft[c, high_epochs, i])
                resp_low = np.mean(resp_dft[c, low_epochs, i])
                resp_high = np.mean(resp_dft[c, high_epochs, i])
                real_low_diff = np.absolute(base_low - resp_low)
                real_high_diff = np.absolute(base_high - resp_low)

                shuffle_low = np.concatenate([base_dft[c, low_epochs, i], resp_dft[c, low_epochs, i]], axis = 0)
                shuffle_high = np.concatenate([base_dft[c, high_epochs, i], resp_dft[c, high_epochs, i]], axis = 0)
                count_low = np.zeros([perm_num])  
                count_high = np.zeros([perm_num]) 
                low_diff = np.zeros([perm_num])
                high_diff = np.zeros([perm_num])

                for k in range(perm_num):
                    np.random.shuffle(shuffle_low)
                    np.random.shuffle(shuffle_high)
                    shuffle_bl, shuffle_rl = np.split(shuffle_low, 2)
                    shuffle_bh, shuffle_rh = np.split(shuffle_high, 2)
                    low_diff[k] = np.absolute(np.mean(shuffle_bl) - np.mean(shuffle_rl))
                    high_diff[k] = np.absolute(np.mean(shuffle_bh) - np.mean(shuffle_rh))
                    if low_diff[k] >= real_low_diff:
                        count_low[k] = 1
                    else: 
                        count_low[k] = 0   
                    if high_diff[k] >= real_high_diff:
                        count_high[k] = 1
                    else:
                        count_high[k] = 0

                low_stats[c, i] = np.sum(count_low)/perm_num 
                high_stats[c, i] = np.sum(count_low)/perm_num 

        #benjamin hochberg correction for multiple comparisons
        print('computing Benjamin Hochberg correction for multiple comparisons')
        low_rank = np.zeros([num_chans, num_freqs])
        high_rank = np.zeros([num_chans, num_freqs])
        for c in range(num_chans):
            low_rank[c, :] = low_stats[c, :].copy().argsort().argsort()
            high_rank[c, :] = high_stats[c, :].copy().argsort().argsort()

        low_rank = low_rank + 1
        high_rank = high_rank + 1
        fdr = 0.25 #false discovery rate as a decimal
        print('number of comparisons/tests adjusted for: ', num_freqs)
        low_c = low_rank * (fdr/num_freqs)
        high_c = high_rank * (fdr/num_freqs)
        low_sig = np.zeros([num_chans, num_freqs], dtype = int)
        high_sig = np.zeros([num_chans, num_freqs], dtype = int)

        for c in range(num_chans):
            for i in range(num_freqs):  
                if low_stats[c, i] <= low_c[c, i]:
                    low_sig[c, i] = 1
                else: 
                    low_sig[c, i] = 0
        for c in range(num_chans):        
            for i in range(num_freqs):
                if high_stats[c, i] <= high_c[c, i]:
                    high_sig[c, i] = 1
                else:
                    high_sig[c, i] = 0

        print(' Number of Freqs with sig differences (between baseline and response segments) in LOW STATE TRIALS')
        for c in range(num_chans):
            print(np.str(all_epochs.ch_names[c]), ': ', np.sum(low_sig[c, :]))

        print(' Number of Freqs with sig differences (between baseline and response segments) in HIGH STATE TRIALS')
        for c in range(num_chans):
            print(np.str(all_epochs.ch_names[c]), ': ', np.sum(high_sig[c, :])) 

        #for each channel, plot power spectra for the 4 conditions
        adj_freqs = np.linspace(start_freq + (freq_res/2), end_freq - (freq_res/2), num = len(dft_freqs)-1 ) 
        fig = plt.figure( figsize = (14,14))
        font = {'size'   : 12}
        plt.rc('font', **font)
        for i in range(num_chans):
            plt.subplot(num_chans,2,(i*2)+1)
            #baseline, low state
            avg_bl = np.mean(base_dft[i, low_epochs, :], axis = 0)
            error_bl = np.std(base_dft[i, low_epochs, :], axis = 0)
            plt.plot(adj_freqs, avg_bl, color = 'g', label = 'Baseline')
            plt.fill_between(adj_freqs, avg_bl - error_bl, avg_bl + error_bl, facecolor = 'g', alpha=0.1)
            #response, low state
            avg_rl = np.mean(resp_dft[i, low_epochs, :], axis = 0)
            error_rl = np.std(resp_dft[i, low_epochs, :], axis = 0)
            plt.plot(adj_freqs, avg_rl, color = 'orange', label = 'Response')
            plt.fill_between(adj_freqs, avg_rl - error_rl, avg_rl + error_rl, facecolor = 'orange', alpha=0.1)
            #sig stars
            sig_freqs_low = adj_freqs[np.nonzero(low_sig[i, :])]
            y_constant = np.zeros([len(sig_freqs_low)]) - 5
            plt.plot(sig_freqs_low, y_constant, color = 'k', marker = "*", linestyle = ":")
            plt.title('Low State: ' + str(all_epochs.ch_names[i]))
            if i == num_chans-1:
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Log Power')
                plt.legend(loc = 'lower left', prop={'size': 10})

            plt.subplot(num_chans,2,(i*2)+2)
            #baseline, high state
            avg_bh = np.mean(base_dft[i, high_epochs, :], axis = 0)
            error_bh = np.std(base_dft[i, high_epochs, :], axis = 0)
            plt.plot(adj_freqs, avg_bh, color = 'b', label = 'Baseline')
            plt.fill_between(adj_freqs, avg_bh - error_bh, avg_bh + error_bh, facecolor = 'b', alpha=0.1)
            #response, high state
            avg_rh = np.mean(resp_dft[i, high_epochs, :], axis = 0)
            error_rh = np.std(resp_dft[i, high_epochs, :], axis = 0)
            plt.plot(adj_freqs, avg_rh, color = 'r', label = 'Response')
            plt.fill_between(adj_freqs, avg_rh - error_rh, avg_rh + error_rh, facecolor = 'r', alpha=0.1)
            #sig stars
            sig_freqs_high = adj_freqs[np.nonzero(high_sig[i, :])]    
            y_constant = np.zeros([len(sig_freqs_high)]) - 5
            plt.plot(sig_freqs_high, y_constant, color = 'k', marker = "*", linestyle = ":")
            plt.title('High State: ' + str(all_epochs.ch_names[i]))
            if i == num_chans-1:
                plt.legend(loc = 'lower left', prop={'size': 10})
        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(wspace=0.15)
        plt.suptitle('Comparison between Baseline and Response Segments ')
        plt.show()          




        #TEST FOR POWER SPECTRAL DIFFERENCES BETWEEN LOW VS HIGH STATES (separately for baseline and response segments)

        base_stats = np.zeros([num_chans, num_freqs])
        resp_stats = np.zeros([num_chans, num_freqs])
        #perm_num = 1000
        print('number of permutations: ', perm_num)

        for c in range(num_chans):
            print('generating stats for ', np.str(all_epochs.ch_names[c]))
            for i in range(num_freqs): 
                base_low = np.mean(base_dft[c, low_epochs, i])
                base_high = np.mean(base_dft[c, high_epochs, i])
                resp_low = np.mean(resp_dft[c, low_epochs, i])
                resp_high = np.mean(resp_dft[c, high_epochs, i])
                real_base_diff = np.absolute(base_low - base_high)
                real_resp_diff = np.absolute(resp_low - resp_high)

                shuffle_base = np.concatenate([base_dft[c, low_epochs, i], base_dft[c, high_epochs, i]], axis = 0)
                shuffle_resp = np.concatenate([resp_dft[c, low_epochs, i], resp_dft[c, high_epochs, i]], axis = 0)
                count_base = np.zeros([perm_num])  
                count_resp = np.zeros([perm_num]) 
                base_diff = np.zeros([perm_num])
                resp_diff = np.zeros([perm_num])

                for k in range(perm_num):
                    np.random.shuffle(shuffle_base)
                    np.random.shuffle(shuffle_resp)
                    shuffle_bl, shuffle_bh = np.split(shuffle_base, 2)
                    shuffle_rl, shuffle_rh = np.split(shuffle_resp, 2)
                    base_diff[k] = np.absolute(np.mean(shuffle_bl) - np.mean(shuffle_bh))
                    resp_diff[k] = np.absolute(np.mean(shuffle_rl) - np.mean(shuffle_rh))
                    if base_diff[k] >= real_base_diff:
                        count_base[k] = 1
                    else: 
                        count_base[k] = 0   
                    if resp_diff[k] >= real_resp_diff:
                        count_resp[k] = 1
                    else:
                        count_resp[k] = 0

                base_stats[c, i] = np.sum(count_base)/perm_num 
                resp_stats[c, i] = np.sum(count_resp)/perm_num 

        #benjamin hochberg correction for multiple comparisons
        print('computing Benjamin Hochberg correction for multiple comparisons')
        base_rank = np.zeros([num_chans, num_freqs])
        resp_rank = np.zeros([num_chans, num_freqs])
        for c in range(num_chans):
            base_rank[c, :] = base_stats[c, :].copy().argsort().argsort()
            resp_rank[c, :] = resp_stats[c, :].copy().argsort().argsort()

        base_rank = base_rank + 1
        resp_rank = resp_rank + 1
        fdr = 0.25 #false discovery rate as a decimal
        num_tests = num_freqs
        print('number of comparisons/tests adjusted for: ', num_tests)
        base_c = base_rank * (fdr/num_tests)
        resp_c = resp_rank * (fdr/num_tests)
        base_sig = np.zeros([num_chans, num_freqs], dtype = int)
        resp_sig = np.zeros([num_chans, num_freqs], dtype = int)

        for c in range(num_chans):
            for i in range(num_freqs):  
                if base_stats[c, i] <= base_c[c, i]:
                    base_sig[c, i] = 1
                else: 
                    base_sig[c, i] = 0
        for c in range(num_chans):        
            for i in range(num_freqs):
                if resp_stats[c, i] <= resp_c[c, i]:
                    resp_sig[c, i] = 1
                else:
                    resp_sig[c, i] = 0

        print(' Number of Freqs with sig differences (between high and low) in BASELINE')
        num_sig_base = np.zeros([num_chans], dtype = int)
        for c in range(num_chans):
            num_sig_base[c] = np.sum(base_sig[c, :], dtype = int)
            print(np.str(all_epochs.ch_names[c]), ': ', num_sig_base[c])       

        print(' Number of Freqs with sig differences (between high and low) in RESPONSE')
        num_sig_resp = np.zeros([num_chans], dtype = int)
        for c in range(num_chans):
            num_sig_resp[c] = np.sum(resp_sig[c, :], dtype = int)
            print(np.str(all_epochs.ch_names[c]), ': ', num_sig_resp[c]) 

        #for each channel, plot power spectra for the 4 conditions
        adj_freqs = np.linspace(start_freq + (freq_res/2), end_freq - (freq_res/2), num = num_freqs ) 
        fig = plt.figure( figsize = (14,14))
        font = {'size'   : 12}
        plt.rc('font', **font)
        for i in range(num_chans):
            plt.subplot(num_chans,2,(i*2)+1)
            #baseline, high state
            avg_bh = np.mean(base_dft[i, high_epochs, :], axis = 0)
            error_bh = np.std(base_dft[i, high_epochs, :], axis = 0)
            plt.plot(adj_freqs, avg_bh, color = 'b', label = 'High State')
            plt.fill_between(adj_freqs, avg_bh - error_bh, avg_bh + error_bh, facecolor = 'b', alpha=0.1)
            #baseline, low state
            avg_bl = np.mean(base_dft[i, low_epochs, :], axis = 0)
            error_bl = np.std(base_dft[i, low_epochs, :], axis = 0)
            plt.plot(adj_freqs, avg_bl, color = 'g', label = 'Low State')
            plt.fill_between(adj_freqs, avg_bl - error_bl, avg_bl + error_bl, facecolor = 'g', alpha=0.1)
            #sig stars
            sig_freqs_base = adj_freqs[np.nonzero(base_sig[i, :])]
            y_constant = np.zeros([len(sig_freqs_base)]) - 5
            plt.plot(sig_freqs_base, y_constant, color = 'k', marker = "*", linestyle = ":")
            plt.title('Baseline: ' + str(all_epochs.ch_names[i]))
            if i == num_chans-1:
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Log Power')
                plt.legend(loc = 'lower left', prop={'size': 10})

            plt.subplot(num_chans,2,(i*2)+2)
            #response, high state
            avg_rh = np.mean(resp_dft[i, high_epochs, :], axis = 0)
            error_rh = np.std(resp_dft[i, high_epochs, :], axis = 0)
            plt.plot(adj_freqs, avg_rh, color = 'r', label = 'High State')
            plt.fill_between(adj_freqs, avg_rh - error_rh, avg_rh + error_rh, facecolor = 'r', alpha=0.1)
            #response, low state
            avg_rl = np.mean(resp_dft[i, low_epochs, :], axis = 0)
            error_rl = np.std(resp_dft[i, low_epochs, :], axis = 0)
            plt.plot(adj_freqs, avg_rl, color = 'orange', label = 'Low State')
            plt.fill_between(adj_freqs, avg_rl - error_rl, avg_rl + error_rl, facecolor = 'orange', alpha=0.1)
            #sig stars
            sig_freqs_resp = adj_freqs[np.nonzero(resp_sig[i, :])]    
            y_constant = np.zeros([len(sig_freqs_resp)]) - 5
            plt.plot(sig_freqs_resp, y_constant, color = 'k', marker = "*", linestyle = ":")
            plt.title('Response: ' + str(all_epochs.ch_names[i]))
            if i == num_chans-1:
                plt.legend(loc = 'lower left', prop={'size': 10})
        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(wspace=0.15)
        plt.suptitle('Comparison between high and low states ')
        plt.show()




                
#-------------------------------------------------------------
# TIME FREQUENCY ANALYSIS
#------------------------------------------------------------
                
    def time_freq_analysis(input_file = 'all_epochs2_GW-epo.fif', high_file = 'high_epochs.npy', low_file =                             'low_epochs.npy', perm_num = 1000):
                
        '''
	This function calculates time-freq representations of the data using fft with a sliding window. TFR plots for
	each channel are generated. Permutation statistics test for event-related desynchronisation in each brain state
	(within-trial, one tailed) and for sig differences in ERD between brain states (between trial, two-tailed).

        INPUT:
            input_file: file containing epoch data - output of clean_epochs
            high_file: saved array with indices of high-state epochs - output of divide_epochs
            low_file: saved array with indices of low-state epochs - output of divide_epochs
            perm_num: num of permutations for non-parametric stats. Above 1000 will sig increase processing time.
        
        OUPUT:
            time-freq graphs for each channel - all epochs
            time-freq graphs for each channel - high and low state epochs
            graph showing time-freq representation of differences between brain states alongside only sig differences
        '''

        #Import modules
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import copy
        import mne                          # available eeg processing toolbox
        import mbp                          # dft processing module from Moritz
        import sys
        from random import randrange
        from itertools import compress
        import mne.time_frequency as tf 

        

        # SHORT 3 SEC EPOCHS - DOES SUB-ALPHA UNDERGO ERD
        #SPLIT EPOCHS FIRST
        import mne.time_frequency as tf 
        alpha_lim = [4, 6.0001]
        freq_res = 2
        #load data
        high_epochs = np.load(high_file)
        low_epochs = np.load(low_file)
        all_epochs = mne.read_epochs(input_file, preload= True)
        all_epochs.baseline = None
        all_epochs.crop(tmin = -16, tmax = 16)

        print('dividing 15s epochs into 3s epochs')
        #divide RESP: 15s epochs into 3s
        resp_1 = all_epochs.copy().crop(tmin = 0, tmax = 3, include_tmax = False)
        resp_2 = all_epochs.copy().crop(tmin = 3, tmax = 6, include_tmax = False)
        resp_2.shift_time(tshift = 0, relative = False)
        resp_3 = all_epochs.copy().crop(tmin = 6, tmax = 9, include_tmax = False)
        resp_3.shift_time(tshift = 0, relative = False)
        resp_4 = all_epochs.copy().crop(tmin = 9, tmax = 12, include_tmax = False)
        resp_4.shift_time(tshift = 0, relative = False)
        resp_5 = all_epochs.copy().crop(tmin = 12, tmax = 15, include_tmax = False)
        resp_5.shift_time(tshift = 0, relative = False)
        epoch_list = [resp_1, resp_2, resp_3, resp_4, resp_5]
        short_resp = mne.concatenate_epochs(epoch_list)

        #divide BASE: 15s epochs into 3s
        base_1 = all_epochs.copy().crop(tmin = -3, tmax = 0, include_tmax = False)
        base_2 = all_epochs.copy().crop(tmin = -6, tmax = -3, include_tmax = False)
        base_2.shift_time(tshift = -3, relative = False)
        base_3 = all_epochs.copy().crop(tmin = -9, tmax = -6, include_tmax = False)
        base_3.shift_time(tshift = -3, relative = False)
        base_4 = all_epochs.copy().crop(tmin = -12, tmax = -9, include_tmax = False)
        base_4.shift_time(tshift = -3, relative = False)
        base_5 = all_epochs.copy().crop(tmin = -15, tmax = -12, include_tmax = False)
        base_5.shift_time(tshift = -3, relative = False)
        epoch_list = [base_1, base_2, base_3, base_4, base_5]
        short_base = mne.concatenate_epochs(epoch_list)
        long_base = all_epochs.copy().crop(tmin = -15, tmax = 0, include_tmax = False) # for baseline correction


        #create numpy array
        np_short_base = mne.Epochs.get_data(short_base)
        np_short_resp = mne.Epochs.get_data(short_resp)
        np_long_base = mne.Epochs.get_data(long_base)
        np_dim = np_short_resp.shape

        #initialise stft analysis
        dft_freqs = np.arange(alpha_lim[0], alpha_lim[1], freq_res)

        # create test stft to calculate size of output arrays based on given parameters
        test_short = mbp.mbp(np_short_resp[:,0,:], fs = 250, freqs = dft_freqs, tstep = 0.1, tlen = 1/freq_res, plot =                              False )
        short_dim = [test_short.shape[1], test_short.shape[3], test_short.shape[0]]
        test_long = mbp.mbp(np_long_base[:,0,:], fs = 250, freqs = dft_freqs, tstep = 0.1, tlen = 1/freq_res, plot = 
                              False )
        long_dim = [test_long.shape[1], test_long.shape[3]]
        print('stft output: The number of frequencies is:', short_dim[0])
        print('stft output: The number of time steps is:', short_dim[1])
        base_stft = np.zeros((np_dim[0], np_dim[1], short_dim[1]))  # empty array (nepochs, nchans, nfreqs, ntimes)  
        resp_stft = np.zeros((np_dim[0], np_dim[1], short_dim[1]))  # empty array (nepochs, nchans, nfreqs, ntimes)
        long_base_stft = np.zeros((test_long.shape[0], np_dim[1], long_dim[1]))  # empty array (nepochs, nchans, nfreqs,                          #ntimes)  
        num_epochs = short_dim[2]
        num_chans = np_dim[1]
        num_freqs = short_dim[0]
        num_times = short_dim[1]
        #del test_stft

        #iterate through egg channels and generate stft
        print('generating stft for short baseline segments')
        for i in range(num_chans):
            base_stft[:,i,:] = np.squeeze(mbp.mbp(np_short_base[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.1, tlen = 
                                            1/freq_res))
        print('generating stft for short response segments')
        for i in range(num_chans):
            resp_stft[:,i,:] = np.squeeze(mbp.mbp(np_short_resp[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.1, tlen = 
                                        1/freq_res))
        print('generating stft for long baseline segments')
        for i in range(num_chans):
            long_base_stft[:,i,:] = np.squeeze(mbp.mbp(np_long_base[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.1, tlen                                    =1/freq_res)) 
                                     
        resp_times = np.linspace(0+(0.5/freq_res), 3-(0.5/freq_res), num = num_times) # output time steps

        avg_base = np.mean(long_base_stft, axis = 2)
        avg_base = np.concatenate((avg_base, avg_base, avg_base, avg_base, avg_base), axis = 0)
        base_array = np.zeros((num_epochs, num_chans, num_times))
        for i in range(num_times):
            base_array[:, :, i] = avg_base
    
        #baseline correction
        base_stft = (base_stft - base_array)/ base_array
        resp_stft = (resp_stft - base_array)/ base_array

        #COMPUTING WITHIN-TRIAL PERMUTATION STATISTICS FOR ALL EPOCHS
        print('computing within-trial permutation statistics for ALL epochs (resp vs base)')
        #perm_num = 500
        print('number of permutations: ', perm_num)
        perm_stat = np.zeros([num_chans, num_times])
        for c in range(num_chans):
            print('generating stats for ', np.str(short_base.ch_names[c]))
            for j in range(num_times):
                avg_resp = np.mean(resp_stft[:, c, j], axis = 0)
                avg_base = np.mean(base_stft[:, c, :]) # average across time and epochs for baseline
                real_diff = avg_base - avg_resp

                shuffle_data = np.concatenate([resp_stft[:, c, j], np.mean(base_stft[:, c, :], axis = 1)],axis=0)
                count = np.zeros([perm_num])                            
                shuffle_diff = np.zeros([perm_num])
                for k in range(perm_num):
                    np.random.shuffle(shuffle_data)
                    shuffle_resp, shuffle_base = np.split(shuffle_data, 2)
                    shuffle_diff[k] = np.mean(shuffle_base) - np.mean(shuffle_resp)
                    if shuffle_diff[k] >= real_diff:
                        count[k] = 1
                    else: 
                        count[k] = 0
                perm_stat[c, j] = np.sum(count)/perm_num 

        #benjamin hochberg correction for multiple comparisons
        print('computing Benjamin Hochberg correction for multiple comparisons')
        rank_array = np.zeros([num_chans, num_times])
        for c in range(num_chans):
            rank_array[c, :] = perm_stat[c, :].copy().ravel().argsort().argsort().reshape(perm_stat[c, :].shape)    
        rank_array = rank_array + 1
        fdr = 0.25 #false discovery rate as a decimal
        num_tests = np.size(rank_array[0, :])
        print('number of comparisons/tests adjusted for: ', num_tests)
        c_value = rank_array * (fdr/num_tests)
        sig_array = np.zeros([num_chans, num_times], dtype = int)

        #generate array indicating significance of each chan-time-freq pixel
        for c in range(num_chans): 
            for j in range(num_times): 
                if perm_stat[c, j] <= c_value[c, j]:
                    sig_array[c, j] = 1
                else: 
                    sig_array[c, j] = 0
        print('number of significant timepoints for ALL epochs:')
        num_sig = np.zeros([num_chans], dtype = int)
        for c in range(num_chans):
            num_sig[c] = np.sum(sig_array[c, :], dtype = int)
            print(np.str(short_base.ch_names[c]), ': ', num_sig[c])

        # PLOT TFR AND SIG ERD FOR ALL EPOCHS
        fig = plt.figure( figsize = (8,14))
        font = {'size'   : 12}
        plt.rc('font', **font)
        for c in range(num_chans):
            plt.subplot(num_chans, 1, c+1)
            avg_alpha = np.mean(resp_stft[:, c, :], axis = 0)
            std_alpha = np.std(resp_stft[:, c, :], axis = 0)
            plt.plot(resp_times, avg_alpha, color = 'r')
            plt.fill_between(resp_times, avg_alpha - std_alpha, avg_alpha + std_alpha, facecolor = 'r',alpha=0.2)
            plt.title(str(short_base.ch_names[c]) + ': Avg Sub-Alpha Power')
            plt.axvline(x= 0, color = 'k')
            plt.xlabel('Time (s)')
            plt.ylabel('% Change')
            sig_times = resp_times[np.nonzero(sig_array[c, :])]
            max_y = np.amax(avg_alpha)
            y_constant = np.zeros([len(sig_times)]) + max_y + (max_y*0.2)
            plt.plot(sig_times, y_constant, color = 'k', marker = "*", linestyle = ":")
            plt.axhline(y = 0, color = 'k', linestyle = '--')
        fig.subplots_adjust(hspace=0.55)
        plt.suptitle('Event-Related Desynchronisation in Sub-Alpha (4-6Hz) Range')



        # 3S ANALYSIS OF BRAIN STATE SPECIFIC TFRS

        freq_lim = [3.5, 13.5]
        freq_res = 1
        #load data
        high_epochs = np.load(high_file)
        low_epochs = np.load(low_file)
        all_epochs = mne.read_epochs(input_file, preload= True)
        all_epochs.baseline = None
        all_epochs.crop(tmin = -18, tmax = 18)
        l_epochs = all_epochs[low_epochs]
        h_epochs = all_epochs[high_epochs]

        #divide LOW RESP: 15s epochs into 3s
        seg_1 = l_epochs.copy().crop(tmin = 0, tmax = 3, include_tmax = False)
        seg_2 = l_epochs.copy().crop(tmin = 3, tmax = 6, include_tmax = False)
        seg_2.shift_time(tshift = 0, relative = False)
        seg_3 = l_epochs.copy().crop(tmin = 6, tmax = 9, include_tmax = False)
        seg_3.shift_time(tshift = 0, relative = False)
        seg_4 = l_epochs.copy().crop(tmin = 9, tmax = 12, include_tmax = False)
        seg_4.shift_time(tshift = 0, relative = False)
        seg_5 = l_epochs.copy().crop(tmin = 12, tmax = 15, include_tmax = False)
        seg_5.shift_time(tshift = 0, relative = False)
        epoch_list = [seg_1, seg_2, seg_3, seg_4, seg_5]
        short_rl = mne.concatenate_epochs(epoch_list)

        #divide LOW BASE: 15s epochs into 3s
        seg_1 = l_epochs.copy().crop(tmin = -3, tmax = 0, include_tmax = False)
        seg_2 = l_epochs.copy().crop(tmin = -6, tmax = -3, include_tmax = False)
        seg_2.shift_time(tshift = -3, relative = False)
        seg_3 = l_epochs.copy().crop(tmin = -9, tmax = -6, include_tmax = False)
        seg_3.shift_time(tshift = -3, relative = False)
        seg_4 = l_epochs.copy().crop(tmin = -12, tmax = -9, include_tmax = False)
        seg_4.shift_time(tshift = -3, relative = False)
        seg_5 = l_epochs.copy().crop(tmin = -15, tmax = -12, include_tmax = False)
        seg_5.shift_time(tshift = -3, relative = False)
        epoch_list = [seg_1, seg_2, seg_3, seg_4, seg_5]
        short_bl = mne.concatenate_epochs(epoch_list)
        long_bl = l_epochs.copy().crop(tmin = -15, tmax = 0, include_tmax = False) # for baseline correction

        #divide HIGH RESP: 15s epochs into 3s
        seg_1 = h_epochs.copy().crop(tmin = 0, tmax = 3, include_tmax = False)
        seg_2 = h_epochs.copy().crop(tmin = 3, tmax = 6, include_tmax = False)
        seg_2.shift_time(tshift = 0, relative = False)
        seg_3 = h_epochs.copy().crop(tmin = 6, tmax = 9, include_tmax = False)
        seg_3.shift_time(tshift = 0, relative = False)
        seg_4 = h_epochs.copy().crop(tmin = 9, tmax = 12, include_tmax = False)
        seg_4.shift_time(tshift = 0, relative = False)
        seg_5 = h_epochs.copy().crop(tmin = 12, tmax = 15, include_tmax = False)
        seg_5.shift_time(tshift = 0, relative = False)
        epoch_list = [seg_1, seg_2, seg_3, seg_4, seg_5]
        short_rh = mne.concatenate_epochs(epoch_list)

        #divide HIGH BASE: 15s epochs into 3s
        seg_1 = h_epochs.copy().crop(tmin = -3, tmax = 0, include_tmax = False)
        seg_2 = h_epochs.copy().crop(tmin = -6, tmax = -3, include_tmax = False)
        seg_2.shift_time(tshift = -3, relative = False)
        seg_3 = h_epochs.copy().crop(tmin = -9, tmax = -6, include_tmax = False)
        seg_3.shift_time(tshift = -3, relative = False)
        seg_4 = h_epochs.copy().crop(tmin = -12, tmax = -9, include_tmax = False)
        seg_4.shift_time(tshift = -3, relative = False)
        seg_5 = h_epochs.copy().crop(tmin = -15, tmax = -12, include_tmax = False)
        seg_5.shift_time(tshift = -3, relative = False)
        epoch_list = [seg_1, seg_2, seg_3, seg_4, seg_5]
        short_bh = mne.concatenate_epochs(epoch_list)
        long_bh = h_epochs.copy().crop(tmin = -15, tmax = 0, include_tmax = False) # for baseline correction


        #create numpy array
        np_short_bl = mne.Epochs.get_data(short_bl)
        np_short_rl = mne.Epochs.get_data(short_rl)
        np_short_bh = mne.Epochs.get_data(short_bh)
        np_short_rh = mne.Epochs.get_data(short_rh)
        np_long_bl = mne.Epochs.get_data(long_bl)
        np_long_bh = mne.Epochs.get_data(long_bh)
        num_chans = np_short_rh.shape[1]

        #initialise stft analysis
        dft_freqs = np.arange(freq_lim[0], freq_lim[1], freq_res)

        # create test stft to calculate size of output arrays based on given parameters
        test_short = mbp.mbp(np_short_rh[:,0,:], fs = 250, freqs = dft_freqs, tstep = 0.05, tlen = 1/freq_res,plot = False)
        num_freqs = test_short.shape[1]
        num_times = test_short.shape[3]
        num_shortepo = test_short.shape[0]
        test_long = mbp.mbp(np_long_bh[:,0,:], fs = 250, freqs = dft_freqs, tstep = 0.05, tlen = 1/freq_res, plot = False )
        num_longtim = test_long.shape[3]
        num_longepo = test_long.shape[0]
        print('stft output: The number of frequencies is:', num_freqs)
        print('stft output: The number of short time steps is: ', num_times)
        print('stft output: The number of long time steps is: ', num_longtim)
        print('num of epochs (long baseline): ', num_longepo)
        print('num of epochs (short segments): ', num_shortepo)
        print('num of chans: ', num_chans )
        bl_stft = np.zeros((num_shortepo, num_chans, num_freqs, num_times))  # empty(nepochs, nchans, nfreqs, ntimes)  
        bh_stft = np.zeros((num_shortepo, num_chans, num_freqs, num_times))  # empty(nepochs, nchans, nfreqs, ntimes)
        rl_stft = np.zeros((num_shortepo, num_chans, num_freqs, num_times))  # empty(nepochs, nchans, nfreqs, ntimes)
        rh_stft = np.zeros((num_shortepo, num_chans, num_freqs, num_times))  # empty(nepochs, nchans, nfreqs, ntimes)
        long_bl_stft = np.zeros((num_longepo, num_chans, num_freqs, num_longtim)) # empty (nepochs, nchans, nfreqs,ntimes)  
        long_bh_stft = np.zeros((num_longepo, num_chans, num_freqs, num_longtim)) # empty (nepochs, nchans, nfreqs,ntimes)  

        #iterate through egg channels and generate stft
        print('generating stft for short LOW baseline segments')
        for i in range(num_chans):
            bl_stft[:,i,:,:] = np.squeeze(mbp.mbp(np_short_bl[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.05, tlen = 
                                            1/freq_res))
        print('generating stft for short HIGH baseline segments')
        for i in range(num_chans):
            bh_stft[:,i,:,:] = np.squeeze(mbp.mbp(np_short_bh[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.05, tlen = 
                                            1/freq_res))
        print('generating stft for short LOW response segments')
        for i in range(num_chans):
            rl_stft[:,i,:,:] = np.squeeze(mbp.mbp(np_short_rl[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.05, tlen = 
                                        1/freq_res))
        print('generating stft for short HIGH response segments')
        for i in range(num_chans):
            rh_stft[:,i,:,:] = np.squeeze(mbp.mbp(np_short_rh[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.05, tlen = 
                                        1/freq_res))
        print('generating stft for long LOW baseline segments')
        for i in range(num_chans):
            long_bl_stft[:,i,:,:] = np.squeeze(mbp.mbp(np_long_bl[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.05, tlen 
                                                     =1/freq_res)) 
        print('generating stft for long HIGH baseline segments')
        for i in range(num_chans):
            long_bh_stft[:,i,:,:] = np.squeeze(mbp.mbp(np_long_bh[:,i,:], fs = 250, freqs = dft_freqs, tstep = 0.05, tlen 
                                                     =1/freq_res))

        resp_times = np.linspace(0+(0.5/freq_res), 3-(0.5/freq_res), num = num_times) # output time steps
        adj_freqs = np.linspace(freq_lim[0] + (freq_res/2), freq_lim[1] - (freq_res/2), num = num_freqs) # output freqs                                             

        # format long baseline array for baseline correction
        avg_base = np.mean(long_bl_stft, axis = 3)
        avg_base = np.concatenate((avg_base, avg_base, avg_base, avg_base, avg_base), axis = 0)
        bl_array = np.zeros((num_shortepo, num_chans, num_freqs, num_times))
        for i in range(num_times):
            bl_array[:, :, :, i] = avg_base
        avg_base = np.mean(long_bh_stft, axis = 3)
        avg_base = np.concatenate((avg_base, avg_base, avg_base, avg_base, avg_base), axis = 0)
        bh_array = np.zeros((num_shortepo, num_chans, num_freqs, num_times))
        for i in range(num_times):
            bh_array[:, :, :, i] = avg_base

        #baseline correction
        bl_stft = (bl_stft - bl_array)/ bl_array
        bh_stft = (bh_stft - bh_array)/ bh_array
        rl_stft = (rl_stft - bl_array)/ bl_array
        rh_stft = (rh_stft - bh_array)/ bh_array


        # BETWEEN-TRIALS PERMUTATION BASED ANALYSIS (DIFFERENCE BETWEEN HIGH AND LOW STATE RESPONSES) 
        #computing permutations statistics
        #perm_num = 1000
        print('computing between-trial permutation statistics (diff between high and low state responses')
        print('number of permutations: ', perm_num)
        perm_stat = np.zeros([num_chans, num_freqs, num_times])
        for c in range(num_chans):
            print('generating stats for ', np.str(all_epochs.ch_names[c]))
            for i in range(num_freqs):
                for j in range(num_times):
                    avg_low = np.mean(rl_stft[:, c, i, j], axis = 0)
                    avg_high = np.mean(rh_stft[:, c, i, j], axis = 0)
                    real_diff = np.absolute(avg_low - avg_high)

                    shuffle_data = np.concatenate([rl_stft[:, c, i, j], rh_stft[:, c, i, j]], axis = 0)
                    count = np.zeros([perm_num])                            
                    shuffle_diff = np.zeros([perm_num])
                    for k in range(perm_num):
                        np.random.shuffle(shuffle_data)
                        shuffle_low, shuffle_high = np.split(shuffle_data, 2)
                        shuffle_diff[k] = np.absolute(np.mean(shuffle_low) - np.mean(shuffle_high))
                        if shuffle_diff[k] >= real_diff:
                            count[k] = 1
                        else: 
                            count[k] = 0
                    perm_stat[c, i, j] = np.sum(count)/perm_num 

        #benjamin hochberg correction for multiple comparisons
        print('computing Benjamin Hochberg correction for multiple comparisons')
        rank_array = np.zeros([num_chans, num_freqs, num_times])
        for c in range(num_chans):
            rank_array[c, :, :] = perm_stat[c, :, :].copy().ravel().argsort().argsort().reshape(perm_stat[c, :, :].shape)    
        rank_array = rank_array + 1
        fdr = 0.25 #false discovery rate as a decimal
        num_tests = np.size(rank_array[0, :, :])
        print('number of comparisons/tests adjusted for: ', num_tests)
        c_value = rank_array * (fdr/num_tests)
        sig_array = np.zeros([num_chans, num_freqs, num_times], dtype = int)

        #generate array indicating significance of each chan-time-freq pixel
        for c in range(num_chans):
            for i in range(num_freqs): 
                for j in range(num_times): 
                    if perm_stat[c, i, j] <= c_value[c, i, j]:
                        sig_array[c, i, j] = 1
                    else: 
                        sig_array[c, i, j] = 0
        print('number of significant time-freq pixels for each channel:')
        num_sig = np.zeros([num_chans], dtype = int)
        for c in range(num_chans):
            num_sig[c] = np.sum(sig_array[c, :, :], dtype = int)
            print(np.str(all_epochs.ch_names[c]), ': ', num_sig[c])

        #calculate time-freq representation of diff between low and high epochs
        diff_data = np.mean(rl_stft, axis = 0) - np.mean(rh_stft, axis = 0)
        diff_data_sig = diff_data * sig_array

        #COMPUTING WITHIN-TRIAL PERMUTATION STATISTICS FOR LOW STATE
        #perm_num = 1000
        print('computing within-trial permutation statistics for LOW STATE (resp vs base)')
        print('number of permutations: ', perm_num)
        perm_stat = np.zeros([num_chans, num_freqs, num_times])
        for c in range(num_chans):
            print('generating stats for ', np.str(all_epochs.ch_names[c]))
            for i in range(num_freqs):
                for j in range(num_times):
                    avg_resp = np.mean(rl_stft[:, c, i, j], axis = 0)
                    avg_base = np.mean(bl_stft[:, c, i, :]) # average across time and epochs for baseline
                    real_diff = avg_base - avg_resp

                    shuffle_data = np.concatenate([rl_stft[:, c, i, j], np.mean(bl_stft[:, c, i, :], axis = 1)], axis = 0)
                    count = np.zeros([perm_num])                            
                    shuffle_diff = np.zeros([perm_num])
                    for k in range(perm_num):
                        np.random.shuffle(shuffle_data)
                        shuffle_resp, shuffle_base = np.split(shuffle_data, 2)
                        shuffle_diff[k] = np.mean(shuffle_base) - np.mean(shuffle_resp)
                        if shuffle_diff[k] >= real_diff:
                            count[k] = 1
                        else: 
                            count[k] = 0
                    perm_stat[c, i, j] = np.sum(count)/perm_num 

        #benjamin hochberg correction for multiple comparisons
        print('computing Benjamin Hochberg correction for multiple comparisons')
        rank_array = np.zeros([num_chans, num_freqs, num_times])
        for c in range(num_chans):
            rank_array[c, :, :] = perm_stat[c, :, :].copy().ravel().argsort().argsort().reshape(perm_stat[c, :, :].shape)    
        rank_array = rank_array + 1
        fdr = 0.25 #false discovery rate as a decimal
        num_tests = np.size(rank_array[0, :, :])
        print('number of comparisons/tests adjusted for: ', num_tests)
        c_value = rank_array * (fdr/num_tests)
        low_sig_array = np.zeros([num_chans, num_freqs, num_times], dtype = int)

        #generate array indicating significance of each chan-time-freq pixel
        for c in range(num_chans):
            for i in range(num_freqs): 
                for j in range(num_times): 
                    if perm_stat[c, i, j] <= c_value[c, i, j]:
                        low_sig_array[c, i, j] = 1
                    else: 
                        low_sig_array[c, i, j] = 0
        print('number of significant time-freq pixels for LOW states:')
        num_sig = np.zeros([num_chans], dtype = int)
        for c in range(num_chans):
            num_sig[c] = np.sum(low_sig_array[c, :, :], dtype = int)
            print(np.str(all_epochs.ch_names[c]), ': ', num_sig[c])

        #COMPUTING WITHIN-TRIAL PERMUTATION STATISTICS FOR HIGH STATE
        print('computing within-trial permutation statistics for HIGH STATE (resp vs base)')
        print('number of permutations: ', perm_num)
        perm_stat = np.zeros([num_chans, num_freqs, num_times])
        for c in range(num_chans):
            print('generating stats for ', np.str(all_epochs.ch_names[c]))
            for i in range(num_freqs):
                for j in range(num_times):
                    avg_resp = np.mean(rh_stft[:, c, i, j], axis = 0)
                    avg_base = np.mean(bh_stft[:, c, i, :]) # average across time and epochs for baseline
                    real_diff = avg_base - avg_resp

                    shuffle_data = np.concatenate([rh_stft[:, c, i, j], np.mean(bh_stft[:, c, i, :], axis = 1)], axis = 0)
                    count = np.zeros([perm_num])                            
                    shuffle_diff = np.zeros([perm_num])
                    for k in range(perm_num):
                        np.random.shuffle(shuffle_data)
                        shuffle_resp, shuffle_base = np.split(shuffle_data, 2)
                        shuffle_diff[k] = np.mean(shuffle_base) - np.mean(shuffle_resp)
                        if shuffle_diff[k] >= real_diff:
                            count[k] = 1
                        else: 
                            count[k] = 0
                    perm_stat[c, i, j] = np.sum(count)/perm_num 

        #benjamin hochberg correction for multiple comparisons
        print('computing Benjamin Hochberg correction for multiple comparisons')
        rank_array = np.zeros([num_chans, num_freqs, num_times])
        for c in range(num_chans):
            rank_array[c, :, :] = perm_stat[c, :, :].copy().ravel().argsort().argsort().reshape(perm_stat[c, :, :].shape)    
        rank_array = rank_array + 1
        fdr = 0.25 #false discovery rate as a decimal
        num_tests = np.size(rank_array[0, :, :])
        print('number of comparisons/tests adjusted for: ', num_tests)
        c_value = rank_array * (fdr/num_tests)
        high_sig_array = np.zeros([num_chans, num_freqs, num_times], dtype = int)

        #generate array indicating significance of each chan-time-freq pixel
        for c in range(num_chans):
            for i in range(num_freqs): 
                for j in range(num_times): 
                    if perm_stat[c, i, j] <= c_value[c, i, j]:
                        high_sig_array[c, i, j] = 1
                    else: 
                        high_sig_array[c, i, j] = 0
        print('number of significant time-freq pixels for HIGH states:')
        num_sig = np.zeros([num_chans], dtype = int)
        for c in range(num_chans):
            num_sig[c] = np.sum(high_sig_array[c, :, :], dtype = int)
            print(np.str(all_epochs.ch_names[c]), ': ', num_sig[c])

        #plot graph showing erd and sig erd for low, high and diff
        fig = plt.figure( figsize = (12,12))
        font = {'size'   : 11}
        plt.rc('font', **font)
        for c in range(num_chans):
            #low resp vs base
            low_avg_data = np.mean(rl_stft[:, c, :, :], axis = 0)
            low_sig_avg = low_avg_data * low_sig_array[c, :, :]
            vmax = np.max(np.abs(low_avg_data))
            vmin = -vmax
            plt.subplot(num_chans, 3, (c*3)+1)
            plt.title(np.str(all_epochs.ch_names[c]) + ': LOW State')
            l_masked = np.ma.masked_where(low_sig_avg == 0, low_sig_avg)
            plt.imshow(low_avg_data, cmap=plt.cm.gray,
                               extent=[resp_times[0], resp_times[-1], adj_freqs[0], adj_freqs[-1]],
                                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax, alpha = 0.8)
            plt.imshow(l_masked, cmap=plt.cm.RdBu_r,
                               extent=[resp_times[0], resp_times[-1], adj_freqs[0], adj_freqs[-1]],
                                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            #plt.axvline(x= 0, color = 'r')
            #if c == num_chans-1:
            plt.colorbar()

            #high resp vs base
            high_avg_data = np.mean(rh_stft[:, c, :, :], axis = 0)
            high_sig_avg = high_avg_data * high_sig_array[c, :, :]
            vmax = np.max(np.abs(high_avg_data))
            vmin = -vmax
            plt.subplot(num_chans, 3, (c*3)+2)
            plt.title(np.str(all_epochs.ch_names[c]) + ': HIGH State')
            h_masked = np.ma.masked_where(high_sig_avg == 0, high_sig_avg)
            plt.imshow(high_avg_data, cmap=plt.cm.gray,
                               extent=[resp_times[0], resp_times[-1], adj_freqs[0], adj_freqs[-1]],
                                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax, alpha = 0.8)
            plt.colorbar()
            plt.imshow(h_masked, cmap=plt.cm.RdBu_r,
                               extent=[resp_times[0], resp_times[-1], adj_freqs[0], adj_freqs[-1]],
                                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            #plt.axvline(x= 0, color = 'r')

            #diff in HIGH RESP VS LOW RESP
            vmax = np.max(np.abs(diff_data))
            vmin = -vmax
            plt.subplot(num_chans, 3, (c*3)+3 )
            plt.title(np.str(all_epochs.ch_names[c]) + ': Diff (LOW-HIGH)')
            d_masked = np.ma.masked_where(diff_data_sig[c,:,:] == 0, diff_data_sig[c,:,:])
            plt.imshow(diff_data[c, :, :], cmap = plt.cm.gray,
                   extent=[resp_times[0], resp_times[-1], adj_freqs[0], adj_freqs[-1]],
                   aspect='auto', origin='lower', vmin=vmin, vmax=vmax, alpha = 0.8)
            plt.imshow(d_masked, cmap=plt.cm.RdBu_r,
                               extent=[resp_times[0], resp_times[-1], adj_freqs[0], adj_freqs[-1]],
                                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')

        fig.subplots_adjust(hspace=0.6)
        fig.subplots_adjust(wspace=0.4)
        plt.show()

