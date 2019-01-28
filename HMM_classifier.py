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
    def __init__(self, n_class=3, n_state=8, n_fea=7, n_iter=50, verbose=True):
        self.n_class=n_class
        self.n_iter = n_iter
        self.models = []
        for i in range(n_class):
            # create a model with 10 hidden states with features.
            model = HMM(n_state, n_fea)
            self.models.append(model)
        
        self.trained = False
        
    def train(self, samples, labels):
        print('Training started')
        for k in range(self.n_class):
            print('model class: {} training...'.format(k))
            self.models[k].fit(samples[:,k], num_runs=self.n_iter)
        print('Training finished')
        self.trained = True
            
    def score_samples(self, samples):
        assert self.trained, 'HMM Models should be trained before prediction.'
        print('scoring...')
        size = samples.shape[0]
        scores = np.zeros((size, self.n_class), dtype=np.float32)
        for k in range(size):
            print('\r{:6.2f}%({}/{})'.format(100*(k+1)/size, k+1, size), end='')
            for i in range(self.n_class):
                scores[k,i] = self.models[i].posterior(samples[k,i])
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
    
    