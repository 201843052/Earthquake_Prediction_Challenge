#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install obspy')
get_ipython().system('pip install numpy==1.20.0')


# In[2]:


import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from obspy import read
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import dask as dd
import random
from scipy.fftpack import dct
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import pandas as pd
import tensorflow as tf
from models.utils import focal_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score
from sklearn.utils import shuffle
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling1D, Flatten, Dropout, Dense,Conv1D

from helper.model import training
from helper.read_file_in_repo import identify_files_in_directory, read_data_from_disk
from helper.window_target_generate import *


# # Reading Files

# In[4]:


signal_paths = identify_files_in_directory("./data/signal")
noise_paths = identify_files_in_directory("./data/noise")

short_noise_paths = noise_paths[9000:10000]
short_signal_paths = signal_paths

raw_data_signal, signal_target, signal_index= read_data_from_disk("./data/signal",short_signal_paths)
raw_data_noise, noise_target, noise_index = read_data_from_disk("./data/noise",short_noise_paths)

total_raw_data = raw_data_noise+raw_data_signal
total_index = noise_index+signal_index

y = np.concatenate((noise_target,signal_target),axis=0)
y_class = [int(1) if tar is not None else int(0) for tar in y  ]
index_ts = np.array(range(np.array(total_raw_data).shape[0])) 


# # Training Model

# In[259]:


pred = training(total_raw_data,total_index,index_ts)


# In[260]:


len(pred)


# In[261]:


pred_1 = [1 if i != None else 0 for i in total_index]
pred_2 = [1 if i != None else 0 for i in pred] 


# In[262]:



print(accuracy_score(pred_1,pred_2),f1_score(pred_1,pred_2)) 


# # Testing on New Data

# In[263]:


short_noise_paths_X = noise_paths[9102:10202]
short_signal_paths_X = signal_paths[1102:1222]


# In[264]:


raw_data_signal_test, signal_target_test, signal_index_test = read_data_from_disk("./data/signal",short_signal_paths_X)
raw_data_noise_test, noise_target_test, noise_index_test = read_data_from_disk("./data/noise",short_noise_paths_X)


# In[265]:


total_raw_data_test = raw_data_noise_test + raw_data_signal_test
total_index_test = noise_index_test + signal_index_test
index_ts_test = np.array(range(np.array(total_raw_data_test).shape[0]))


# In[275]:


def testing(total_raw_data,total_index,chosen_ix_train):
    rolling_window_index_a,rolling_window_index_b,rolling_window_index_c = rolling_window_index()
    window_A = get_windowA(rolling_window_index_a,total_raw_data)
    ts_ix,chosen_index_a,pred_label = windowA_pred(window_A)  # need to check this
    window_B,a_ix = get_windowB(ts_ix,chosen_index_a,pred_label,window_A,rolling_window_index_b,window_A)
    b_ix = windowB_pred(window_B)
    window_C = get_windowC(b_ix,window_B,rolling_window_index_c)
    c_ix = windowC_pred(window_C) # Need to Check
    predictions = final_pred(pred_label,a_ix,b_ix,c_ix,rolling_window_index_c,total_raw_data,chosen_ix_train)
    return predictions


# In[276]:


predictions = testing(total_raw_data_test,total_index_test,index_ts_test)


# In[277]:


pred_1 = [1 if i != None else 0 for i in total_index_test]
pred_2 = [1 if i != None else 0 for i in predictions] 


# In[278]:


from sklearn.metrics import accuracy_score,f1_score
print(accuracy_score(pred_1,pred_2),f1_score(pred_1,pred_2)) 


# In[279]:


len(pred)


# In[280]:


len(predictions)


# In[273]:




