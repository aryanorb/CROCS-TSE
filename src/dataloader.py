# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:09:22 2024

@author: gist
"""

# https://github.com/JorisCos/LibriMix?tab=readme-ov-file#Features
# https://pytorch.org/audio/main/generated/torchaudio.datasets.LibriMix.html

import torch
import torchaudio
import librosa
import random
import numpy as np
import os

from torchaudio.datasets import LibriMix
from torch.utils.data import Dataset

def _collate_fn(batch):
    
    mixture_segments, target_segments, ref_segments, target_speaker_index = zip(*batch)
    
    # tuple to batch samples 
    mixture_segments            = torch.stack(mixture_segments) 
    target_segments             = torch.stack(target_segments)
    target_speaker_index        = torch.tensor(target_speaker_index)
    
    # ref_segments is variable lenghts.
    max_len = max([ref.shape[-1] for ref in ref_segments])
    
    batch_size = mixture_segments.shape[0]
    
    new_ref_segments = torch.zeros(batch_size, 1, max_len)
    
    for i, ref in enumerate(ref_segments):
        
        if ref.shape[-1] < max_len:
            shortage = max_len - ref.shape[-1]
            
            ref_data = ref.squeeze(0).numpy()
            
            new_ref = np.pad(ref_data, (0, shortage), 'wrap')

            new_ref = torch.Tensor(new_ref)
            
        else:
            new_ref = ref
        new_ref_segments[i] = new_ref

    return mixture_segments, target_segments, new_ref_segments, target_speaker_index

class LibriMixDevTestSet(Dataset):
    def __init__(self, data_list, root, subset, task, enrollment_lengths,  sample_rate = 8000, num_speakers = 2, mode = 'min'):
        super(LibriMixDevTestSet, self).__init__()
        
        print('## Dev-Test dataset || sr: {}, mode: {}'.format(sample_rate, mode))
        
        self.data_list      = data_list
        self.root           = root
        self.subset         = subset
        self.task           = task
        self.num_speakers   = num_speakers
        self.sample_rate    = sample_rate
        self.mode           = mode
        
        if enrollment_lengths is None:
            self.enrollment_lengths = None
        else:
            self.enrollment_lengths = int(enrollment_lengths * self.sample_rate)
        
        self.test_type          = self.data_list.split('_')[1] # test or dev
        self.data_path          = os.path.join(self.root, 'Libri{}Mix/wav{}k/{}/{}'.format(num_speakers, int(self.sample_rate // 1000), self.mode, self.test_type))
        
        self.mixture_list       = []
        self.target_list        = []
        self.enrollment_list    = []

        lines = open(self.data_list).read().splitlines()
        for index, line in enumerate(lines):
            mixture_path, target_speaker_id, enrollment_path = line.split()
            
            # mixture
            mixture_path = os.path.join(self.data_path, self.task, mixture_path+'.wav').replace('\\','/')
            self.mixture_list.append(mixture_path)
            
            # target
            if index % 2 == 0:
                destination = 's1'
                target_path = mixture_path.replace(self.task, destination)
                self.target_list.append(target_path)
            elif index % 2 == 1:
                destination = 's2'
                target_path = mixture_path.replace(self.task, destination)
                self.target_list.append(target_path)
    
            # enrollment
            enrollment_path = os.path.join(self.data_path, enrollment_path+'.wav').replace('\\','/')
            self.enrollment_list.append(enrollment_path)
            
            # print('index', index)
            # print('mixture:',mixture_path)
            # print('target:',target_path)
            # print('enrollment:',enrollment_path)
            # print('\n')
            
    def _get_audio_path(self, index):
        return self.mixture_list[index], self.target_list[index], self.enrollment_list[index]
    
    def _get_target_audio_file_name(self, index):
        path = self.target_list[index]
        return path.strip().replace('\\', '/').split('/')[-2] + '_' + path.strip().split('/')[-1]    
            
    def __len__(self):
        return len(self.mixture_list)
    
    def __getitem__(self, index):
        
        mixture_path = self.mixture_list[index]
        mixture, _ = librosa.core.load(mixture_path, sr = self.sample_rate)
  
        target_path = self.target_list[index]
        target, _ = librosa.core.load(target_path, sr = self.sample_rate)
        
        assert mixture.shape[0] == target.shape[0], 'Error: check your data: {} {}'.format(mixture.shape[0], target.shape[0])
        
        enrollment_path = self.enrollment_list[index]
        enrollment, _ = librosa.core.load(enrollment_path, sr = self.sample_rate)

        if self.enrollment_lengths is None:
            enrollment = enrollment
        else:
            if enrollment.shape[0] <= self.enrollment_lengths:
                shortage     = int(self.enrollment_lengths - enrollment.shape[0])
                enrollment   = np.pad(enrollment, (0,shortage), 'wrap')
            
            enroll_index    = np.int64(random.random()*(enrollment.shape[0]-self.enrollment_lengths))
            enrollment      = enrollment[enroll_index:enroll_index + self.enrollment_lengths]
                    
        mixture_std     = np.std(mixture)
        target_std      = np.std(target)
        enrollment_std  = np.std(enrollment)
        
        # 
        mixture     = mixture / mixture_std
        #target      = target / target_std 
        enrollment  = enrollment / enrollment_std
        
        mixture    = torch.FloatTensor(mixture).unsqueeze(dim=0)
        target     = torch.FloatTensor(target).unsqueeze(dim=0)
        enrollment = torch.FloatTensor(enrollment).unsqueeze(dim=0)
        
        return mixture, target, enrollment
        

class LibriMixTrainSet(Dataset):
    def __init__(self, data_list, root, subset, task,  train_lengths = None, ref_lengths = None, sample_rate = 8000, num_speakers = 2, mode = 'min'):
        super(LibriMixTrainSet, self).__init__()

        print('## Train dataset || sr: {}, mode: {}'.format(sample_rate, mode))
        
        self.data_list      = data_list
        self.root           = root
        self.subset         = subset
        self.task           = task
        self.num_speakers   = num_speakers
        self.sample_rate    = sample_rate
        self.mode           = mode
    
        if train_lengths is None:
            self.train_lengths = int(4.0 * self.sample_rate)
        else:
            self.train_lengths = int(train_lengths * self.sample_rate)
            
        if ref_lengths is None:
            self.ref_lengths = ref_lengths
        else:
            self.ref_lengths = int(ref_lengths * self.sample_rate)
            
        self.data_path = os.path.join(self.root, 'Libri{}Mix'.format(num_speakers)).replace('\\','/')
        
        
        self.LibriMixData = LibriMix(
            root            = self.root, 
            subset          = self.subset, 
            task            = self.task,
            num_speakers    = self.num_speakers,
            sample_rate     = self.sample_rate,
            mode            = self.mode
        )
        
        lines = open(self.data_list).read().splitlines()
        
        self.labels_to_path = {}
        self.speaker_labels = []

        for path in lines:
            target_source   = path.split('/')[-2]
            
            if target_source == 's1':
                speaker_label = path.split('/')[-1].split('-')[0]
                self.speaker_labels.append(speaker_label)
                
                if speaker_label not in self.labels_to_path:
                    self.labels_to_path[speaker_label] = []
                if speaker_label in path:
                    self.labels_to_path[speaker_label].append(path)
                    
            elif target_source == 's2':
                speaker_label = path.split('/')[-1].split('_')[1].split('-')[0]
                self.speaker_labels.append(speaker_label)

                if speaker_label not in self.labels_to_path:
                    self.labels_to_path[speaker_label] = []
                if speaker_label in path:
                   self.labels_to_path[speaker_label].append(path)
        
        # sorting based on the nubmer 
        self.speaker_labels = sorted(set(self.speaker_labels), key = int)

        # {'19': 0, '26': 1, '27': 2, '27': 3 ...}
        self.speaker_to_index = {key :index for index, key in enumerate(self.speaker_labels)}
        
        #print(self.speaker_to_index)
    
    def get_number_of_speakers(self):
        return len(self.speaker_labels)
    
    def __len__(self):
        return len(self.LibriMixData)
    
    def __getitem__(self, index):
        
        sr, mixture_path, source_paths = self.LibriMixData.get_metadata(index)
        
        #print(sr, mixture_path, source_paths)
        
        number = random.randint(0, 1)
        
        if number == 0: # source 1
            target_path     = source_paths[number].replace('\\','/')
            speaker_label   = target_path.split('/')[4].split('-')[0]
     
            reference_path  = target_path
            while reference_path == target_path:
                reference_path = np.random.choice(self.labels_to_path[speaker_label])
        else: # source 2
            target_path     = source_paths[number].replace('\\','/')
            speaker_label   = target_path.split('/')[4].split('-')[2].split('_')[1]
            
            reference_path  = target_path
            while reference_path == target_path:
                reference_path = np.random.choice(self.labels_to_path[speaker_label])
        
        mixture_path = os.path.join(self.data_path, mixture_path).replace('\\','/')
        target_path  = os.path.join(self.data_path, target_path).replace('\\','/')
        
        # print('random number: {}, label: {}'.format(number, speaker_label))
        
        # print('mixture_path:\t{}'.format(mixture_path))
        # print('target_path :\t{}'.format(target_path))
        # print('ref_path    :\t{}'.format(reference_path))
            
        mixture, _   = librosa.core.load(mixture_path, sr = self.sample_rate)
        target, _    = librosa.core.load(target_path, sr = self.sample_rate)
        reference, _ = librosa.core.load(reference_path, sr = self.sample_rate)
        
        assert mixture.shape[0] == target.shape[0], 'Error: check your data: {} {}'.format(mixture.shape[0], target.shape[0])
        
        # Normalizing using std
        mixture_std     = np.std(mixture)
        target_std      = np.std(target)
        reference_std   = np.std(reference)
        
        mixture     = mixture / mixture_std
        #target      = target / target_std
        reference   = reference / reference_std
        
        if mixture.shape[0] <= self.train_lengths:
            shortage    = int(self.train_lengths - mixture.shape[0])
            mixture     = np.pad(mixture, (0,shortage), 'wrap')
            target      = np.pad(target, (0,shortage), 'wrap')
        
        start_index      = np.int64(random.random()*(mixture.shape[0]-self.train_lengths))
        mixture_segments = mixture[start_index:start_index + self.train_lengths]
        target_segments  = target[start_index:start_index + self.train_lengths]
        
        # None is to utilizing full_lengths 
        if self.ref_lengths is None:
            ref_segments    = reference
        else:
            if reference.shape[0] <= self.ref_lengths:
                shortage    = int(self.ref_lengths - reference.shape[0])
                reference   = np.pad(reference, (0,shortage), 'wrap')
            
            ref_index       = np.int64(random.random()*(reference.shape[0]-self.ref_lengths))
            ref_segments    = reference[ref_index:ref_index + self.ref_lengths]
        
        #print(mixture_segments.shape[0], target_segments.shape[0], ref_segments.shape[0])
        
        mixture_segments    = torch.FloatTensor(mixture_segments).unsqueeze(dim=0)
        target_segments     = torch.FloatTensor(target_segments).unsqueeze(dim=0)
        ref_segments        = torch.FloatTensor(ref_segments).unsqueeze(dim=0)
        
        target_speaker_index = self.speaker_to_index[speaker_label]
        
        #print(target_speaker_index)
        
        return mixture_segments, target_segments, ref_segments, target_speaker_index
            
if __name__ == '__main__':
    
    # mix_clean = clean
    # mix_both = noisy
    
    
    #data_list = './libri2mix_train_list.txt'
    data_list = './libri2mix_train_list_8k.txt'
    path  = '/data3/data/LibriMix'
    subset = 'train-100'
    task  = 'sep_noisy' # sep_noisy
    sample_rate = 8000
    ref_lengths = None
    train_lengths = 4.0
    
    train_dataset = LibriMixTrainSet(data_list, path, subset, task, train_lengths, ref_lengths, sample_rate)
    
    dev_dataset = LibriMixDevTestSet(
        data_list = './speakerbeam_dev_mixture2enrollment.txt', 
        root = path,
        subset= 'train-100',
        task = 'mix_clean',
        enrollment_lengths = None,
        sample_rate = sample_rate
    )
    
    test_dataset = LibriMixDevTestSet(
        data_list = './speakerbeam_test_mixture2enrollment.txt', 
        root = path,
        subset= 'test',
        task = 'mix_both',
        enrollment_lengths = None,
        sample_rate = sample_rate
    )
    mixture, target, enrollment = train_dataset[1]
    print('train:', mixture.shape, target.shape, enrollment.shape)
    mixture, target, enrollment = test_dataset[3]
    print('test:', mixture.shape, target.shape, enrollment.shape)
    
    #print(train_dataset[9], len(train_dataset))
    
