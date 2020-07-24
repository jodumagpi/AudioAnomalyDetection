from scipy.io import wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import time
import soundfile as sf
from datetime import timedelta as td
import scipy

class Denoiser:
    """
    Denoiser class.
    
        Denoiser Args:
        
            n_grad_freq (int): number of frequency channels to smooth over with the mask
            n_grad_time (int): number of time channels to smooth over with the mask
            n_fft (int): number audio of frames between STFT columns.
            win_len (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
            hop_len (int):number audio of frames between STFT columns.
            n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
            prop_dec (float): Extent should you decrease noise (1 = all, 0 = none)
            verbose (bool): Set verbosity
            visual (bool): Whether to plot the steps of the algorithm

        """
    def __init__(self,
        n_grad_freq=2,
        n_grad_time=4,
        n_fft=2048,
        win_len=2048,
        hop_len=512,
        n_std_thresh=1.5,
        prop_dec=0.8,
        verbose=False,
        visual=False):
        
        self.n_grad_freq = n_grad_freq
        self.n_grad_time = n_grad_time
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_std_thresh = n_std_thresh
        self.prop_dec = prop_dec
        self.verbose = verbose
        self.visual = visual
        
    def stft(self, y, n_fft, hop_len, win_len):
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)

    def istft(self, y, hop_len, win_len):
        return librosa.istft(y, hop_len, win_len)

    def amp_to_db(self, x):
        return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

    def db_to_amp(self, x):
        return librosa.core.db_to_amplitude(x, ref=1.0)
    
    def plot_spectrogram(self, signal, title):
        fig, ax = plt.subplots(figsize=(20, 4))
        cax = ax.matshow(
            signal,
            origin="lower",
            aspect="auto",
            cmap=plt.cm.seismic,
            vmin=-1 * np.max(np.abs(signal)),
            vmax=np.max(np.abs(signal)),
        )
        fig.colorbar(cax)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_statistics_and_filter(self, mean_freq_noise, std_freq_noise, 
            noise_thresh, smoothing_filter):
        
        fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
        plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
        plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
        plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
        ax[0].set_title("Threshold for mask")
        ax[0].legend()
        cax = ax[1].matshow(smoothing_filter, origin="lower")
        fig.colorbar(cax)
        ax[1].set_title("Filter for smoothing Mask")
        plt.show()
    
    def denoise(self, audio_clip, noise_clip):
        """Remove noise from audio based upon a clip containing only noise

        Args:
            audio_clip (array): Input audio
            noise_clip (array): Input noise
            
            Returns:
            array: The recovered signal with noise subtracted
        """
        verbose = self.verbose
        visual = self.visual
        
        # convert stereo to mono
        self.audio_clip = librosa.to_mono(audio_clip.astype(np.float))
        self.noise_clip = librosa.to_mono(noise_clip.astype(np.float))
        if verbose:
            print('Converting stereo data to mono data.')

        if verbose:
            start = time.time()
        # STFT over noise
        noise_stft = self.stft(self.noise_clip, self.n_fft, self.hop_len, self.win_len)
        # convert to dB
        noise_stft_db = self.amp_to_db(np.abs(noise_stft)) 
        # Calculate statistics over noise
        mean_freq_noise = np.mean(noise_stft_db, axis=1)
        std_freq_noise = np.std(noise_stft_db, axis=1)
        noise_thresh = mean_freq_noise + std_freq_noise * self.n_std_thresh
        if verbose:
            print("STFT on noise:", td(seconds=time.time() - start))
            start = time.time()
        # STFT over signal
        if verbose:
            start = time.time()
        sig_stft = self.stft(self.audio_clip, self.n_fft, self.hop_len, self.win_len)
        sig_stft_db = self.amp_to_db(np.abs(sig_stft))
        if verbose:
            print("STFT on signal:", td(seconds=time.time() - start))
            start = time.time()
        # Calculate value to mask dB to
        mask_gain_dB = np.min(self.amp_to_db(np.abs(sig_stft)))
        if verbose:
            print(noise_thresh, mask_gain_dB)
        # Create a smoothing filter for the mask in time and frequency
        smoothing_filter = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, self.n_grad_freq + 1, endpoint=False),
                    np.linspace(1, 0, self.n_grad_freq + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, self.n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, self.n_grad_time + 2),
                ]
            )[1:-1],
        )
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
        # calculate the threshold for each frequency/time bin
        db_thresh = np.repeat(
            np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
            np.shape(sig_stft_db)[1],
            axis=0,
        ).T
        # mask if the signal is above the threshold
        sig_mask = sig_stft_db < db_thresh
        if verbose:
            print("Masking:", td(seconds=time.time() - start))
            start = time.time()
        # convolve the mask with a smoothing filter
        sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
        sig_mask = sig_mask * self.prop_dec
        if verbose:
            print("Mask convolution:", td(seconds=time.time() - start))
            start = time.time()
        # mask the signal
        sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
        )  # mask real
        sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
        sig_stft_amp = (self.db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
            1j * sig_imag_masked
        )
        if verbose:
            print("Mask application:", td(seconds=time.time() - start))
            start = time.time()
        # recover the signal
        recovered_signal = self.istft(sig_stft_amp, self.hop_len, self.win_len)
        recovered_spec = self.amp_to_db(
            np.abs(self.stft(recovered_signal, self.n_fft, self.hop_len, self.win_len))
        )
        if verbose:
            print("Signal recovery:", td(seconds=time.time() - start))
        if visual:
            self.plot_spectrogram(noise_stft_db, title="Noise")
            self.plot_statistics_and_filter(
                mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
            )
            self.plot_spectrogram(sig_stft_db, title="Signal")
            self.plot_spectrogram(sig_mask, title="Mask applied")
            self.plot_spectrogram(sig_stft_db_masked, title="Masked signal")
            self.plot_spectrogram(recovered_spec, title="Recovered spectrogram")
        return recovered_signal

class DownSampler:
    """
        Downsamples the signal to a desired frequency.
        
        Args:
        
        sr (int) : target sample rate
        
        Returns:
        array: downsampled signal
    """
    
    def __init__(self, sr):
        self.sr = sr
        
    def downsample(self, y, orig_sr):.
        y = librosa.to_mono(y.astype(np.float))
        return librosa.resample(y=y, orig_sr=orig_sr, target_sr=self.sr)
