import librosa
import pandas as pd
import numpy as np
import os
import sklearn

class FeatureExtractor:
	"""
		Extract features such as spectral centroid, spectral rolloff, spectral bandwidth, spectral flatness, rmse, MFCC and chroma feature.
	
		FeatureExtractor Args:
	
		sr (int): sampling rate
	
		Returns:
		array: extracted features
	"""

	def __init__(self):
		self.extracted_features = []
		
	def spectral_centroid(self, x, sr, hop_len):
		return librosa.feature.spectral_centroid(x, sr=sr)

	def spectral_rolloff(self, x, sr, hop_len):
		return librosa.feature.spectral_rolloff(x, sr=sr)

	def spectral_bandwidth(self, x, sr, hop_len):
		return librosa.feature.spectral_bandwidth(x, sr=sr)

	def spectral_flatness(self, x, sr, hop_len):
		return librosa.feature.spectral_flatness(x)
	
	def rmse(self, x, sr, hop_len):
		try:
			return librosa.feature.rmse(x, hop_length=hop_len, center=True)
		except:
			return  librosa.feature.rms(x, hop_length=hop_len, center=True)
	
	def mfccs(self, x, sr, hop_len):
		return librosa.feature.mfcc(x, sr=sr)
	
	def chromas(self, x, sr, hop_len):
		return librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_len)
	
	def features(self, x, sr, hop_len):
		self.extracted_features.extend(self.spectral_centroid(x, sr, hop_len))
		self.extracted_features.extend(self.spectral_rolloff(x, sr, hop_len))
		self.extracted_features.extend(self.spectral_bandwidth(x, sr, hop_len))
		self.extracted_features.extend(self.spectral_flatness(x, sr, hop_len))
		self.extracted_features.extend(self.rmse(x, sr, hop_len))
		self.extracted_features.extend(self.mfccs(x, sr, hop_len))
		self.extracted_features.extend(self.chromas(x, sr, hop_len))
	
		return np.asarray(self.extracted_features)	