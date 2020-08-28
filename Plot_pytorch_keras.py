# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:12:30 2020

@author: Qi Chen
"""

import Siamese_CNN_Network as SCN
import pandas as pd

PYTORCH = pd.read_csv('./dataset/embeddings_dist.txt', sep='\t', header=None)
KERAS_df = pd.read_csv('./dataset/pytorch_embeddings_dist.txt', sep='\t', header=None)

SCN._myplot(PYTORCH, KERAS_df, 'pytorch-keras.png', 'Pytorch', 'Keras')
