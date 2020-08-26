# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:49:30 2020

@author: Qi Chen
"""

# set random seed
import random
random.seed(1)

# import other packages
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import r2_score

# setup the environment
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# bp map
bp_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

# class constant
class Constant():
    def __init__(self):
        self.SEQ_LEN = 465
        self.NUM_TRAINING_PAIRS = 20 * 500
        self.EMBEDDING_LEN = 120
        self.NUM_EPOCH = 4
        self.LEARNING_RATE = 1e-4
        self.BATCH_SIZE = 1
        self.NUM_EVAL_N = 500
        self.INPUT_SEQ_PATH = './dataset/pair_shuffle.fa'
        self.INPUT_DIS_PATH = './dataset/dist_shuffle.txt'
        self.OUTPUT_DIS_PATH = './dataset/embeddings_dist.txt'
        self.EVAL_DIS_PATH = './dataset/eval_dist.txt'
        self.EMBEDDINGS_PATH = './dataset/output_embeddings.txt'
        self.PLOT_PATH = './dataset/plot.png'

# class siamese network dataset
class SiameseNetworkDataset():
    def __init__(self, data_fp, target_fp, N, seq_length):
        self.data_fp = data_fp
        self.target_fp = target_fp
        self.N = N
        self.seq_number = 0
        self.seq_length = seq_length
        self.data_tensor = self.gen_data_tensor()
        self.target_tensor = self.gen_target_tensor()
        
    def gen_data_tensor(self):
        seq1 = np.zeros((self.N, self.seq_length, 4))
        seq2 = np.zeros((self.N, self.seq_length, 4))
        cnt = 0
        with open(self.data_fp) as f:
            while True:
                next_n = list(itertools.islice(f, 4))
                if not next_n:
                    break
                if cnt >= self.N:
                    break
                read1 = next_n[1].strip()
                read2 = next_n[3].strip()
                for i, c in enumerate(read1):
                    seq1[cnt, i, bp_map.get(c, 0)] = 1.0
                for i, c in enumerate(read2):
                    seq2[cnt, i, bp_map.get(c, 0)] = 1.0
                cnt += 1
        return [seq1, seq2]

    def gen_target_tensor(self):
        target = np.zeros(self.N)
        with open(self.target_fp) as f:
            for i, line in enumerate(f):
                if i >= self.N:
                    break
                pair_id, dist = line.strip().split()
                target[i] = float(dist)
        return target

# class siamese network
class SiameseNetwork():
    def __init__(self, SiameseDataset, Constant):
        self.Constant = Constant
        self.SiameseDataset = SiameseDataset
        # Share the weight
        self.conv1 = keras.layers.Conv1D(filters=16, kernel_size=5, padding='same', activation='relu', name='conv1')
        self.maxp1 = keras.layers.MaxPooling1D(pool_size=2, name='maxp1')
        self.conv2 = keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', name='conv2')
        self.maxp2 = keras.layers.MaxPooling1D(pool_size=2, name='maxp2')
        self.conv3 = keras.layers.Conv1D(filters=48, kernel_size=5, padding='same', activation='relu', name='conv3')
        self.maxp3 = keras.layers.MaxPooling1D(pool_size=2, name='maxp3')
        self.flatten1 = keras.layers.Flatten(name='flatten1')
        self.dense1 = keras.layers.Dense(self.Constant.EMBEDDING_LEN, name='dense1')

    def forward_one_side(self, network_input):
        CONV1 = self.conv1(network_input)
        MAXP1 = self.maxp1(CONV1)
        CONV2 = self.conv2(MAXP1)
        MAXP2 = self.maxp2(CONV2)
        CONV3 = self.conv3(MAXP2)
        MAXP3 = self.maxp3(CONV3)
        FLAT1 = self.flatten1(MAXP3)
        OUT = self.dense1(FLAT1)
        return OUT
    
    def ContrastiveLoss(self, y_pre, y_align):
        weight_arr = np.diag([1.0 for i in range(self.Constant.BATCH_SIZE)])
        weight_tensor = K.variable(weight_arr)
        loss_contrastive = K.mean(K.dot(weight_tensor, K.pow(y_pre - y_align, 2)))
        return loss_contrastive
    
    def jaccard_dist(self, embedding_list):
        maxout = K.maximum(embedding_list[0], embedding_list[1])
        minout = K.minimum(embedding_list[0], embedding_list[1])
        return 1 - K.sum(minout, axis=1)/K.sum(maxout, axis=1)
    
    def Siamese_model(self):
        input1 = keras.layers.Input(shape=(self.SiameseDataset.seq_length, 4), name='Seq1_Input')
        input2 = keras.layers.Input(shape=(self.SiameseDataset.seq_length, 4), name='Seq2_Input')
        embedding1 = self.forward_one_side(input1)
        embedding2 = self.forward_one_side(input2)
        dist = keras.layers.Lambda(self.jaccard_dist, output_shape=(1,), name='jaccard_dist')([embedding1, embedding2])
        model = keras.models.Model(inputs=[input1, input2], outputs=dist)
        adam_optimizer = keras.optimizers.Adam(lr=self.Constant.LEARNING_RATE)
        model.compile(loss=self.ContrastiveLoss, optimizer=adam_optimizer)
        return model

    def Single_model(self):
        input1 = keras.layers.Input(shape=(self.SiameseDataset.seq_length, 4), name='Seq1_Input')
        embedding1 = self.forward_one_side(input1)
        model = keras.models.Model(inputs=input1, outputs=embedding1)
        return model

# write predict distance
def _writedist(model, fp, constant):
    seq = np.zeros((constant.NUM_EVAL_N, constant.SEQ_LEN, 4))
    cnt = 0
    seq_ids = []
    with open(fp) as f:
        while True:
            next_n = list(itertools.islice(f, 2))
            if not next_n:
                break
            seq_id = next_n[0].strip()[1:]
            read = next_n[1].strip()
            seq_ids.append(seq_id)
            for i, c in enumerate(read):
                seq[cnt, i, bp_map.get(c, 0)] = 1.0
            cnt += 1
    embeddings = model.predict(seq)
    embeddings.tofile(constant.EMBEDDINGS_PATH, sep=',', format='%.4e')
    with open(constant.OUTPUT_DIS_PATH, 'w') as fo:
        for i in range(constant.NUM_EVAL_N):
            for j in range(constant.NUM_EVAL_N):
                if i < j:
                    fo.write('{}-{}\t{:.4f}\n'.format(
                        seq_ids[i], seq_ids[j],
                        _jaccarddist(embeddings[i],
                                     embeddings[j])))

# jaccard dist function
def _jaccarddist(embedding1, embedding2):
    return 1 - np.sum(np.minimum(embedding1, embedding2)) / np.sum(np.maximum(embedding1, embedding2)) 

# plot function
def _myplot(align_dist_df, x_dist_df, save_fp):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.tick_params(axis='both', which='major', labelsize=15)
    hb = ax.hexbin(align_dist_df[1], x_dist_df[1], 
                   gridsize=200, bins='log', cmap='Blues', extent=(0, 1, 0, 1),
                   vmin=1, vmax=4)
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'r')
    ax.set_xlabel('alignment distance', fontsize=20)
    ax.set_ylabel('SENSE', fontsize=20)
    ax.set_title('R2 score: {}'.format(r2_score(align_dist_df[1], x_dist_df[1])))
    
    cbar_ax = fig.add_axes([0.95, 0.1, 0.05, 0.8])
    cbar_ax.tick_params(axis='both', which='major', labelsize=15)
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.set_label('log10(count + 1)', fontsize=20)
    fig.savefig(save_fp, bbox_inches='tight')

# main function
def _main():
    constant = Constant()
    dataset = SiameseNetworkDataset(constant.INPUT_SEQ_PATH, constant.INPUT_DIS_PATH, constant.NUM_TRAINING_PAIRS, constant.SEQ_LEN)
    siamese_network = SiameseNetwork(dataset, constant)
    model = siamese_network.Siamese_model()
    model.summary()
    model.fit(dataset.data_tensor, dataset.target_tensor, epochs=constant.NUM_EPOCH, batch_size=constant.BATCH_SIZE)
    embeddings_model = siamese_network.Single_model()
    _writedist(embeddings_model, './dataset/eval.fa', constant)
    nw_df = pd.read_csv(constant.EVAL_DIS_PATH, sep='\t', header=None)
    my_df = pd.read_csv(constant.OUTPUT_DIS_PATH, sep='\t', header=None)
    _myplot(nw_df, my_df, constant.PLOT_PATH)
    model.save("model.h5")

# main
_main()
