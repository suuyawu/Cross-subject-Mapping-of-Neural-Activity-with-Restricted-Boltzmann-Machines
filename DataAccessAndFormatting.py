
import numpy as np

from scipy.io import loadmat

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

########################################################################################################################################################################################################################################################################################

def DataFormatting (path, moth_database, moth, neurons, stimuli):
    
    data = loadmat(f'{path}{moth_database[0][moth]}_{moth_database[1][moth]}_AnnotatedToShare_v3.mat')
    AdditionalVars = loadmat(f'{path}{moth_database[0][moth]}_{moth_database[1][moth]}_AdditionalInfo.mat')
    
    No_trials = AdditionalVars['No_trials_perNeuronStimulus'].tolist()
    
    spike_times_ALL = [None] * len(neurons)
    targets_ALL = [None] * len(neurons)
    
    for neuron in neurons:

        spike_times_perNEURON = [None] * len(stimuli)
        targets_perNEURON = [None] * len(stimuli)

        for stimulus in stimuli:
            
            spike_times = data['{}_times'.format(neuron)][0,stimulus]
            
            if AdditionalVars['MissingData'][neurons.index(neuron),stimuli.index(stimulus)] == 0:
                
                spike_times_perNEURON[stimuli.index(stimulus)] = spike_times
                targets_perNEURON[stimuli.index(stimulus)] = stimulus*np.ones((spike_times.shape[0],1))
                
            elif AdditionalVars['MissingData'][neurons.index(neuron),stimuli.index(stimulus)] == 1:
                
                spike_times_perNEURON[stimuli.index(stimulus)] = np.empty((AdditionalVars['No_trials'][stimuli.index(stimulus),1],1))
                spike_times_perNEURON[stimuli.index(stimulus)][:] = np.NaN
                targets_perNEURON[stimuli.index(stimulus)] = stimulus*np.ones((AdditionalVars['No_trials'][stimuli.index(stimulus),1],1))
        
            print('(Moth-{}, Neuron-{}, stim-{}) In: orig-{} / Out: targ-{}({})'.format(moth+1, neuron, stimulus, spike_times_perNEURON[stimuli.index(stimulus)].shape, targets_perNEURON[stimuli.index(stimulus)].shape, targets_perNEURON[stimuli.index(stimulus)][0,0]))

        print(125*'-')
        spike_times_ALL[neurons.index(neuron)] = spike_times_perNEURON
        targets_ALL[neurons.index(neuron)] = targets_perNEURON
    
    return spike_times_ALL, targets_ALL, No_trials

########################################################################################################################################################################################################################################################################################

def GaussianKernelFiltering ( spike_times_ALL, targets_ALL, No_trials, neurons, stimuli, dur, f_s, sigma ):
    
    time, T = np.arange(0,dur,1/f_s), int(dur*f_s)
    gauss_kernel_response_ALL = [None] * len(neurons)
    spike_counts_ALL = [None] * len(neurons)

    for neuron in neurons:

        gauss_kernel_response_perNEURON = [None] * len(stimuli)
        spike_counts_perNEURON = [None] * len(stimuli)

        for stimulus in stimuli:

            spike_times_neuron_stimulus = spike_times_ALL[neurons.index(neuron)][stimuli.index(stimulus)]
            gauss_kernel_response = np.zeros( ( spike_times_neuron_stimulus.shape[0], T ) )
            spike_counts = np.zeros( ( spike_times_neuron_stimulus.shape[0], 1 ) )

            for trial in range(spike_times_neuron_stimulus.shape[0]):
                for i in range(spike_times_neuron_stimulus.shape[1]):
                    if ~np.isnan(spike_times_neuron_stimulus[trial,i]):
                        if spike_times_neuron_stimulus[trial,i] <= (dur*1e3):
                            gauss_kernel_response[trial,:] += np.exp( - ((time - 1e-3*spike_times_neuron_stimulus[trial,i])*(time - 1e-3*spike_times_neuron_stimulus[trial,i]))/(2*sigma*sigma) )
                            spike_counts[trial,:] += 1

            gauss_kernel_response_perNEURON[stimuli.index(stimulus)] = gauss_kernel_response
            spike_counts_perNEURON[stimuli.index(stimulus)] = spike_counts

        gauss_kernel_response_ALL[neurons.index(neuron)] = gauss_kernel_response_perNEURON
        spike_counts_ALL[neurons.index(neuron)] = spike_counts_perNEURON
    
    
    signals = np.empty((No_trials,len(neurons),T))
    signals_maxvalpos = np.empty((No_trials,len(neurons),2))
    counts = np.empty((No_trials,len(neurons),1))
    targets = np.empty((No_trials,1))

    for neuron in neurons:
        idx = 0
        for stimulus in stimuli:
            count = 0
            for trial in range( targets_ALL[neurons.index(neuron)][stimuli.index(stimulus)].shape[0] ):
                signals[idx,neurons.index(neuron),:] = gauss_kernel_response_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,:]
                signals_maxvalpos[idx,neurons.index(neuron),0] = np.argmax( gauss_kernel_response_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,:] )
                signals_maxvalpos[idx,neurons.index(neuron),1] = gauss_kernel_response_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,int(signals_maxvalpos[idx,neurons.index(neuron),0])]
                counts[idx,neurons.index(neuron),:] = spike_counts_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,:]
                targets[idx,:] = targets_ALL[neurons.index(neuron)][stimuli.index(stimulus)][count,:]
                idx, count = idx+1, count+1
    
    return signals, signals_maxvalpos, counts, targets
