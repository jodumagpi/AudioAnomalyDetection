import torch
from torch.utils.data import TensorDataset, DataLoader
from sound_processor import DownSampler, Denoiser
from scipy.io import wavfile
import pandas as pd
import numpy as np
import math
import os
import random
import IPython
import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()

class SoundLoader:
    """
    Loads the train, val, and test dataset for sound-related tasks.
    
    Args:
        root_dir (path): Path to the root directory
        data_dir (path): Path to the CSV file
        noise_dir (path): Path to the noise profile 
        sample_rate (int): Frequency used when downsampling is enabled
        sample_per_label (int): Number of times to sample from a label
        sample_len (int): Fixed length of the audio sample in seconds
        val (bool): If validation set is required
        data_split (list or array): Percentage of split for training, validation and testing (train, val, test)
        train_batch (int): Batch size of the training set
        loader (string): Type of loader ['keras' or 'pytorch']
		_downsample_ (bool): Whether to downsample the input
		_denoise_ (bool): Whether to apply noise reduction to the input
		extractors (list[func]): List of functions you want to use as feature extractors
        shuffle (bool): Whether to shuffle the training set
        seed (int): Seed for random processes
        
    Returns:
        data_loader (dictionary): dataloaders for training, validation and testing
    
    """
    
    def __init__(self, root_dir, data_dir, noise_dir,
                 sample_rate=22050,
                 sample_per_label=1,
                 sample_len=1, 
                 val=True, 
                 data_split=[60, 20, 20],
                 train_batch = 32,
                 loader='pytorch',
				 _downsample_ = True,
				 _denoise_ = True,
				 extractors=None,
                 shuffle=True,
                 seed=555):
        
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.noise_dir = noise_dir
        self._downsample_ = _downsample_
        self._denoise_ = _denoise_
        self.data = pd.read_csv(os.path.join(self.root_dir, self.data_dir))
        self.sample_rate = sample_rate
        self.sample_per_label = sample_per_label
        self.sample_len = sample_len
        self.VAL_FLAG = val
        assert sum(data_split) == 100 and sum(data_split)/100 == 1, 'Total of split must be 100 or 1.'
        if self.VAL_FLAG == True: 
            assert len(data_split) == 3, 'Array must have a length of 3 but got {} instead.'.format(len(data_split))
            self.train_split = data_split[0]/sum(data_split)
            self.val_split = data_split[1]/sum(data_split)
            self.test_split = data_split[2]/sum(data_split)
        else:
            assert len(data_split) == 2, 'Array must have a length of 2 but got {} instead.'.format(len(data_split))
            self.train_split = data_split[0]/sum(data_split)
            self.test_split = data_split[2]/sum(data_split)
            
        self.train_batch = train_batch
        self.loader = loader
        self.extractors = extractors
        self.shuffle = shuffle
        self.seed = seed
        
        # manipulators
        self.noise_rate, self.noise = wavfile.read(os.path.join(self.root_dir, self.noise_dir))
        self.noise = self.noise.astype(np.float32)
        self.downsampler = DownSampler(self.sample_rate)
        self.denoiser = Denoiser()
        
        # data
        self.dataset = {'inputs':[], 'labels':[]}
        
    def data_maker(self):
        """
            Segment audio data. 
            Do preprocessing and feature extraction.
        """
        random.seed(self.seed)
        filenames = np.unique(self.data['filename'].values) # get unique filenames
        for file in filenames:
            data_rate, cur_data = wavfile.read(os.path.join(self.root_dir, file))
            cur_data = cur_data.T.astype(np.float32)
            _sample_rate = data_rate
            # denoise
            if self._denoise_:
                delta = len(cur_data[0])/data_rate
                data_rate, cur_data = self.denoiser.denoise(cur_data, self.noise.T, delta)
            # downsample
            if self._downsample_:
                cur_data = self.downsampler.downsample(cur_data, data_rate)
                _sample_rate = self.sample_rate
			
            annos = self.data[self.data['filename'] == file]
            idxs = [] # indices of positive samples
            
            for i in range(len(annos)):
                start, end = math.modf(annos.iloc[i]['start']), math.modf(annos.iloc[i]['end'])
                start, end = int(start[0]*100 + start[1]*60), int(end[0]*100 + end[1]*60) + 1
                
                # trim out positive samples (negatives remain)
                t1, t2 = start * _sample_rate, end * _sample_rate
                idxs.append(np.arange(t1, t2, 1))
                
                # get samples for positives
                dur = np.arange(start, end, 1)[:-1*self.sample_len] # range of values to sample from
                for s in range(self.sample_per_label):
                    if len(dur) >= self.sample_len:
                        t1 = random.sample(list(dur), 1)[0]
                        t2 = t1 + self.sample_len
                        t1, t2 = int(t1 * _sample_rate), int(t2 * _sample_rate)
                        x = cur_data[t1:t2]                    
                        self.dataset['inputs'].append(self.extract_features(x))
                        self.dataset['labels'].append(annos.iloc[i]['label'])
                    else:
                        x = []
                        pad_times = 1 + int(self.sample_len/(end-start)) + int((self.sample_len%(end-start)))
                        for j in range(pad_times):
                            t1, t2 = int(start * _sample_rate), int(end * _sample_rate)
                            x.append(cur_data[t1:t2])
                        x = np.hstack(x)
                        x = x[:self.sample_len * _sample_rate]
                        self.dataset['inputs'].append(self.extract_features(x))
                        self.dataset['labels'].append(annos.iloc[i]['label'])
            
            # get samples for negatives
            idxs = np.hstack(idxs) 
            no_pos = np.delete(cur_data, idxs) # delete positive samples
            idxs = np.argwhere(np.abs(no_pos) < 15) # indices for silent sounds
            neg_data = np.delete(no_pos, idxs) # delete silent sounds
            idxs = random.sample(list(range(len(neg_data))), len(annos)) # index for sampling negative sounds
            for idx in idxs:
                t1 = idx
                t2 = t1 + _sample_rate * self.sample_len

                x = neg_data[t1:t2]
                if len(x) < (_sample_rate*self.sample_len):
                    pass
                else:
                    self.dataset['inputs'].append(self.extract_features(x))
                    self.dataset['labels'].append(np.array([0], dtype=np.int64)[0])
        
        return self.dataset
    
    def extract_features(self, x):
		
        feats = []
        if self.extractors:       
            for e in self.extractors:
                feats.extend(e(x=x, sr=self.sample_rate, hop_len=512))
            feats = np.asarray(feats)
        else:
            feats = x
        return feats
			
    def data_loader(self):
        """
            Create dataloaders for pytorch or keras.
        """
        self.data_maker() #make the dataset
        inputs, labels = np.array(self.dataset['inputs']), np.array(self.dataset['labels'])
        idxs = list(range(len(inputs)))
        random.seed(self.seed)
        if self.shuffle: random.shuffle(idxs)
        if self.VAL_FLAG:
            train = inputs[idxs[:int(len(inputs)*self.train_split)]]
            val = inputs[idxs[int(len(inputs)*self.train_split):int(len(inputs)*(self.train_split+self.val_split))]]
            test = inputs[idxs[int(len(inputs)*(self.train_split+self.val_split)):]]
            train_label = labels[idxs[:int(len(labels)*self.train_split)]]
            val_label = labels[idxs[int(len(labels)*self.train_split):int(len(labels)*(self.train_split+self.val_split))]]
            test_label = labels[idxs[int(len(labels)*(self.train_split+self.val_split)):]]
            
            if self.loader == 'pytorch':
                train_set = TensorDataset(torch.from_numpy(train), torch.from_numpy(train_label))
                val_set = TensorDataset(torch.from_numpy(val), torch.from_numpy(val_label))
                test_set = TensorDataset(torch.from_numpy(test), torch.from_numpy(test_label))

                data_loader = {'train': DataLoader(train_set, batch_size=self.train_batch, shuffle=self.shuffle),
                              'val': DataLoader(val_set),
                              'test': DataLoader(test_set)}
            else:
                train_set = tf.data.Dataset.from_tensor_slices((train, train_label))
                val_set = tf.data.Dataset.from_tensor_slices((val, val_label))
                test_set = tf.data.Dataset.from_tensor_slices((test, test_label))

                data_loader = {'train': train_set.batch(self.train_batch),
                               'val': val_set.batch(1),
                              'test': test_set.batch(1)}

        else:
            train = inputs[idxs[:int(len(inputs)*self.train_split)]]
            test = inputs[idxs[int(len(inputs)*self.train_split):]]
            train_label = labels[idxs[:int(len(labels)*self.train_split)]]
            test_label = labels[idxs[int(len(labels)*(self.train_split+self.val_split)):]]
                        
            if self.loader == 'pytorch':
                
                train_set = TensorDataset(torch.from_numpy(train), torch.from_numpy(train_label))
                test_set = TensorDataset(torch.from_numpy(test), torch.from_numpy(test_label))

                data_loader = {'train': DataLoader(train_set, batch_size=self.train_batch, shuffle=self.shuffle),
                              'test': DataLoader(test_set)}
            else:
                train_set = tf.data.Dataset.from_tensor_slices((train, train_label))
                test_set = tf.data.Dataset.from_tensor_slices((test, test_label))

                data_loader = {'train': train_set.batch(self.train_batch),
                              'test': test_set.batch(1)}
            
        return data_loader