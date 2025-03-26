import numpy as np
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt

def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter
    
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    y = filtfilt(b, a, sig)
    return y

def butter_bandpass_batch(sig_list, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter (batch version)
    # signals are in the sig_list

    y_list = []
    
    for sig in sig_list:
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        y_list.append(y)
    return np.array(y_list)

def hr_fft(sig, fs, harmonics_removal=False):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr


def hr_fft_remove_noise(sig, noise, fs, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig, noise = sig.reshape(-1), noise.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    noise = noise * signal.windows.hann(noise.shape[0])
    
    sig_f = np.abs(fft(sig) - fft(noise))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr


def hr_fft_batch(sig_list, fs, harmonics_removal=True):
    # get heart rate by FFT (batch version)
    # return both heart rate and PSD

    hr_list = []
    for sig in sig_list:
        sig = sig.reshape(-1)
        sig = sig * signal.windows.hann(sig.shape[0])
        sig_f = np.abs(fft(sig))
        low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
        high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
        sig_f_original = sig_f.copy()
        
        sig_f[:low_idx] = 0
        sig_f[high_idx:] = 0

        peak_idx, _ = signal.find_peaks(sig_f)
        sort_idx = np.argsort(sig_f[peak_idx])
        sort_idx = sort_idx[::-1]

        peak_idx1 = peak_idx[sort_idx[0]]
        peak_idx2 = peak_idx[sort_idx[1]]

        f_hr1 = peak_idx1 / sig.shape[0] * fs
        hr1 = f_hr1 * 60

        f_hr2 = peak_idx2 / sig.shape[0] * fs
        hr2 = f_hr2 * 60
        if harmonics_removal:
            if np.abs(hr1-2*hr2)<10:
                hr = hr2
            else:
                hr = hr1
        else:
            hr = hr1

        # x_hr = np.arange(len(sig))/len(sig)*fs*60
        hr_list.append(hr)
    return np.array(hr_list)

def normalize(x):
    return (x-x.mean())/x.std()





from scipy.interpolate import Akima1DInterpolator

def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)
        signal = np.pad(signal, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant')
    freqs = np.fft.fftfreq(len(signal), 1 / Fs) * 60  # in bpm
    ps = np.abs(np.fft.fft(signal))**2
    cutoff = len(freqs)//2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps


def predict_heart_rate(signal, Fs, min_hr=40., max_hr=240., method='fast_ideal_bimodal_filter'):

    if method == 'ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        cs = Akima1DInterpolator(freqs, ps)
        max_val = -np.Inf
        interval = 0.1
        min_bound = max(min(freqs), min_hr)
        max_bound = min(max(freqs), max_hr) + interval
        for bpm in np.arange(min_bound, max_bound, interval):
            cur_val = cs(bpm)
            if cur_val > max_val:
                max_val = cur_val
                max_bpm = bpm
        return max_bpm

    elif method == 'fast_ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        if 0 < max_ind < len(ps)-1:
            inds = [-1, 0, 1] + max_ind
            x = ps[inds]
            f = freqs[inds]
            d1 = x[1]-x[0]
            d2 = x[1]-x[2]
            offset = (1 - min(d1,d2)/max(d1,d2)) * (f[1]-f[0])
            if d2 > d1:
                offset *= -1
            max_bpm = f[1] + offset
        elif max_ind == 0:
            x0, x1 = ps[0], ps[1]
            f0, f1 = freqs[0], freqs[1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        elif max_ind == len(ps) - 1:
            x0, x1 = ps[-2], ps[-1]
            f0, f1 = freqs[-2], freqs[-1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        return max_bpm

    elif method == 'fast_ideal_bimodal_filter':
        """ Same as above but check for secondary peak around 1/2 of first
        (to break the tie in case of occasional bimodal PS)
        Note - this may make metrics worse if the power spectrum is relatively flat
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        max_freq = freqs[max_ind]
        max_ps = ps[max_ind]

        # check for a second lower peak at 0.45-0.55f and >50% power
        freqs_valid = np.logical_and(freqs >= max_freq * 0.45, freqs <= max_freq * 0.55)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        if len(freqs) > 0:
            max_ind_lower = np.argmax(ps)
            max_freq_lower = freqs[max_ind_lower]
            max_ps_lower = ps[max_ind_lower]
        else:
            max_ps_lower = 0

        if max_ps_lower / max_ps > 0.50:
            return max_freq_lower
        else:
            return max_freq
    else:
        raise NotImplementedError 
    
    

def predict_heart_rate_batch(signals, fs, min_hr=40., max_hr=240., method='fast_ideal'):
    return np.array([predict_heart_rate(s, fs, min_hr, max_hr, method) for s in signals])


    
# For ECG signal
from util import detect_peaks

def estimate_average_heartrate(s, sampling_frequency):
    '''Estimates the average heart rate taking as base the input signal and its
    sampling frequency.

    This method will use :py:func:`detect_peaks` to figure out the
    peaks in the noisy signal defined at ``s``.

    Returns:

        float: The estimated average heart-rate in beats-per-minute
        numpy.ndarray: A 1D numpy.ndarray with the peak points

    '''

    min_distance = 60 * sampling_frequency / 120 #BPM

    peaks = detect_peaks(s, mpd=min_distance)
    instantaneous_rates = (sampling_frequency * 60) / np.diff(peaks)

    # remove instantaneous rates which are lower than 30, higher than 240
    selector = (instantaneous_rates>30) & (instantaneous_rates<240)
    return float(np.nan_to_num(instantaneous_rates[selector].mean())), peaks


def estimate_average_heartrate_batch(signals, fs):
    return np.array([estimate_average_heartrate(s, fs)[0] for s in signals])