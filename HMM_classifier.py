#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:13:50 2019

@author: anilosmantur
"""
import numpy as np
from hmm import HMM
import matplotlib.pyplot as plt

np.random.seed(42)

class HMM_classifier():
    ''' classifier with HMM models '''
    def __init__(self, n_class=3, n_state=8, n_fea=7,
                 stop_tol=0.001, min_variance=0.1, num_runs=1
                 max_iter=100, batch_size=None, verbose=True, hmm_type='fully-connected'):
        #hmm types (fully-connected, left-to-right, cyclic)
        self.n_class=n_class
        self.max_iter = max_iter
        self.stop_tol = stop_tol
        self.min_var = min_variance
        self.batch_size = batch_size,
        self.n_runs = num_runs
        self.models = []
        for i in range(n_class):
            # create a model with 10 hidden states with features.
            model = HMM(n_state, n_fea, hmm_type=hmm_type)
            self.models.append(model)
        
        self.trained = False
        
    def train(self, samples, labels):
        print('Training started')
        for label in range(self.n_class):
            print('model class: {} training...'.format(k))
            idx_label = labels == label
            self.models[label].fit(samples[idx_label],
                                   max_steps=self.max_iter,
                                   batch_size=self.batch_size,
                                   TOL=self.stop_tol,
                                   min_var=self.min_var,
                                   num_runs=self.n_runs)
        print('Training finished')
        self.trained = True
            
    def score_samples(self, samples):
        assert self.trained, 'HMM Models should be trained before prediction.'
        print('scoring...')
        size = samples.shape[0]
        scores = np.zeros((size, self.n_class), dtype=np.float32)
        for k in range(self.n_class):
            print('\r{:6.2f}%({}/{})'.format(100*(k+1)/self.n_class, k+1, self.n_class), end='')
            scores[:,i] = self.models[i].posterior(samples)
        print('\nscoring finished.')
        return scores
    
    def predict_samples(self, samples):
        scores = self.score_samples(samples)
        pred = np.argmax(scores, axis=-1)
        return pred
    
    def calculate_accuracy(self, pred, labels):
        size = labels.shape[0]
        corr = (pred == labels).sum()

        acc = corr / size * 100
        print('model acc: {:6.2f}%({}/{})'.format(acc, corr, size))
        
        return acc
    
if __name__ == '__main__':
    n_fea = 7
    n_state = 8
    root_dir = '../data'

    n_sample = 20
    n_class = 3
    seq = 40
    dataset = np.zeros((n_sample, n_class, seq, n_fea), dtype=np.float32) 
    labels = np.zeros((n_sample,), dtype=np.int) 
    for i in range(n_class):
        dataset[:,i] = np.random.randn(n_sample, seq, n_fea).astype(np.float32)
        labels[:] = i
    print('trainset created.')
        
    n_sample = 5
    test_dataset = np.zeros((n_sample, n_class, seq, n_fea), dtype=np.float32)
    t_labels = np.zeros((n_sample,), dtype=np.float32) 
    for i in range(n_class):
        test_dataset[:,i] = np.random.randn(n_sample, seq, n_fea).astype(np.float32)
        t_labels[:] = i
    print('testset created.')
    
    HMM_class = HMM_classifier(n_class=n_class, n_state=n_state, n_fea=n_fea, n_iter=3, verbose=True)
    HMM_class.train(dataset, labels)

    print('train acc:')
    color_pred = HMM_class.predict_samples(dataset)
    HMM_class.calculate_accuracy(color_pred, labels)
    
    print('test acc:')
    color_pred = HMM_class.predict_samples(test_dataset)
    HMM_class.calculate_accuracy(color_pred, t_labels)
    
    