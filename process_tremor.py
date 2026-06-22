import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib
import datetime
import obspy
import seaborn as sns
import math
import datetime
from obspy import UTCDateTime
from obspy.clients.filesystem.sds import Client
from multiprocessing import Pool
import statsmodels as sm
import gc
from pathlib import Path

def main():
    # Example for July 2024
    t_start = datetime.datetime(2023,8,3)
    t_end = datetime.datetime(2024,7,30)
    days = (t_end - t_start).days - 4
    files = [i for i in range(days) if not Path(f"results_all_z_new/{i}.npz").exists()]
    print(files)
    possible_dates = [t_start + datetime.timedelta(days = i) for i in files]
    with Pool() as p:
        p.map(est_signal, possible_dates)
# # Path to the data !!! needs to be adjusted !!!

# Extract glacier hydraulic tremor signal from seismometer data with strong calving events.

def est_signal(t):
    print(t)
    client = Client('QJT_data')
    t_start = t
    t_end = t + datetime.timedelta(days = 1)
    # Times when valid seismometer data starts
    start_dict = {'QJT01': datetime.datetime(2023,8,11,19),
                'QJT02': datetime.datetime(2023,8,14,19),
                'QJT03': datetime.datetime(2023,8,2,16)}

    # Only looking at the vertical component

    # Frequency range in which subglacial tremor is suspected (may be 3-30Hz)
    freqbounds = [(1, 4), (4, 7), (7, 10)]
    stations = ['QJT01', 'QJT02', 'QJT03']
    # components = ['E', 'N', 'Z']
    component = 'Z'
    GHT_all = np.zeros((3, 3, 144))

    for idx1, freq_range in enumerate(freqbounds):
        for idx2, sta in enumerate(stations):
            # for idx3, component in enumerate(components):
            print("\n", freq_range, sta, component)
            t_list = []  # timestamps
            min_list = []  # integrated power in frequency band used for minimum search
            Pxx_min_list = []  # entire PSD at minimum

            # Set start time to the very beginning
            t_start_loop = t_start
            while t_start_loop<t_end:
                print(t_start_loop, end='\r')

                # Read data
                st = client.get_waveforms('XH', sta, '', '*'+component,
                                            UTCDateTime(t_start_loop),
                                            UTCDateTime(t_start_loop+datetime.timedelta(days=1)))
                st.merge(fill_value=0)  # fills data gaps and makes handling easier
                if len(st)>0 and st[0].stats.npts>1:  # only do the analysis if data is present

                    # Remove data when people were present at station
                    if st[0].stats.starttime.date == start_dict[sta].date():
                        st.trim(UTCDateTime(start_dict[sta]), st[0].stats.endtime)

                    # Calculate spectrogram
                    Fs = st[0].stats.sampling_rate
                    NFFT = 5*Fs  # 5s bins for FFT
                    Pxx, freqs, bins, _ = plt.specgram(st[0].data, NFFT=int(NFFT), Fs=Fs, noverlap=0)
                    plt.close()

                    # Select relevant data for minimum power detection
                    idx_low = (np.abs(freqs - freq_range[0])).argmin()
                    idx_high = (np.abs(freqs - freq_range[1])).argmin()
                    Pxx_cut = Pxx[idx_low:idx_high]
                    freqs_cut = freqs[idx_low:idx_high]

                    # Integrate over selected frequency band
                    Pxx_int = np.sum(Pxx_cut, axis=0)

                    # Define chunk size in bins
                    chunk_size = int(10*60 / (NFFT/Fs))  # 10min

                    # Calculate how much padding is needed (if len(bins)%chunk_size!=0)
                    padding_length = chunk_size - (len(Pxx_int) % chunk_size)
                    if padding_length != chunk_size:  # Only pad if necessary
                        Pxx_int = np.pad(Pxx_int, (0, padding_length), constant_values=np.nan)
                        bins = np.pad(bins, (0, padding_length), constant_values=np.nan)
                        Pxx = np.pad(Pxx, pad_width=((0, 0), (0, padding_length)), constant_values=np.nan)

                    # Reshape into chunks
                    reshaped_data = Pxx_int.reshape(-1, chunk_size)
                    reshaped_bins = bins.reshape(-1, chunk_size)
                    Pxx = Pxx.reshape(Pxx.shape[0], int(Pxx.shape[1]/chunk_size), -1).transpose(1,0,2)

                    # Calculate the minimum of each chunk
                    min_values = np.nanmin(reshaped_data, axis=1)
                    min_idx = np.argmin(reshaped_data, axis=1)

                    # Get PSD with minimum power
                    Pxx_select = Pxx[np.arange(Pxx.shape[0]),:,min_idx].T

                    # Fake timestamp of chunk (center of bin)
                    t_stamps = [(st[0].stats.starttime+t).datetime for t in np.nanmean(reshaped_bins, axis=1)]

                    # Append to lists
                    min_list.append(min_values)
                    t_list.append(t_stamps)
                    Pxx_min_list.append(Pxx_select)

                # Continue in while loop
                t_start_loop+=datetime.timedelta(days=1)

            # Convert list of arrays to array
            t_all = np.array(t_list)
            min_all = np.array(min_list)
            # Pxx_all = np.array(Pxx_min_list) # axis = 1?

            # Save results for each station into dictionary
            # GHT_dict[(freqmin, freqmax)] = {'t': t_all,
            #                                 'GHT': min_all}
            try:
                GHT_all[idx1, idx2, :] = min_all
            except:
                GHT_all[idx1, idx2, :] = np.full(shape=144, fill_value=np.nan, dtype=float)
            # GHT_dict[(freqmin, freqmax)] = {'t': t_all,
            #                                 'GHT': min_all,
            #                                 'PSD': Pxx_all} #saves entire PSD
    np.savez_compressed("results_all_z_new/"+ str((t_start - datetime.datetime(2023,8,3)).days), times = t_all, GHT = GHT_all)
    # TODO: delete as much as possible to save memory
    del client
    del t_start
    del t_end
    del start_dict
    del freqbounds
    del stations
    del component
    del GHT_all
    del t_list
    del min_list
    del Pxx_min_list
    del t_start_loop
    del st
    del Fs
    del NFFT
    del Pxx
    del freqs
    del bins
    del idx_low
    del idx_high
    del Pxx_cut
    del freqs_cut
    del Pxx_int
    del chunk_size
    del padding_length
    del reshaped_data
    del reshaped_bins
    del min_values
    del min_idx
    del Pxx_select
    del t_stamps
    del t_all
    del min_all
    # del Pxx_all
    gc.collect()

if __name__ == '__main__':
    main()
