#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def rolling_window_index():
    rolling_window_index_a = []
    window_size = 90
    for a in range(0,1000-window_size,3):
        rolling_window_index_a.append((a,a+window_size))
    rolling_window_index_a = np.array(rolling_window_index_a)
    rolling_window_index_b = []
    window_size=45
    for wind_ix in rolling_window_index_a:
         wind_a_b = []
         for b in range(wind_ix[0],wind_ix[1]-window_size,3):
             wind_a_b.append((b,b+window_size))
         rolling_window_index_b.append(wind_a_b)
    rolling_window_index_b = np.array(rolling_window_index_b)
    rolling_window_index_c = []
    window_size=25
    for ix , b in enumerate(rolling_window_index_b):
        wind_a_b_c = []
        for wind_ix_b in b:
            wind_a_b = []
            for c in range(wind_ix_b[0],wind_ix_b[1]-window_size,3):
                wind_a_b.append((c,c+window_size))
            wind_a_b_c.append(wind_a_b)
        rolling_window_index_c.append(wind_a_b_c)
    rolling_window_index_c = np.array(rolling_window_index_c)
    return (rolling_window_index_a,rolling_window_index_b,rolling_window_index_c)

def windowA_target(total_index,rolling_window_index_a):
    y_train_window_bool = []
    for ix in total_index:
         if ix is None:
             y_train_window_bool.append(np.zeros((304)))
         else:
              y_train_window_bool.append(np.array([1 if ix>wind_ix[0] and ix<wind_ix[1] else 0 for wind_ix in rolling_window_index_a]))
    y_train_window_bool = np.array(y_train_window_bool)
    return y_train_window_bool

def windowB_target(ts_ix,chosen_ix_train,total_index,a_ix,rolling_window_index_b):
    y_train_window_b_bool = []
    for count, row in enumerate(ts_ix):
        row_bool = []
        pwave_index = total_index[chosen_ix_train[row]]
        if pwave_index is None:
           y_train_window_b_bool.append(np.zeros(15))
        else:
             y_train_window_b_bool.append(np.array([1 if pwave_index>wind_ix[0] and pwave_index<wind_ix[1] else 0 for wind_ix in rolling_window_index_b[a_ix[count]]]))
    y_train_window_b_bool = np.array(y_train_window_b_bool)
    return y_train_window_b_bool

def windowC_target(ts_ix,chosen_ix_train,total_index,a_ix,b_ix,rolling_window_index_c):
    y_train_window_c_bool = []
    for count, row in enumerate(ts_ix):
        row_bool = []
        pwave_index = total_index[chosen_ix_train[row]]
        if pwave_index is None:
           y_train_window_c_bool.append(np.zeros(7))
        else:
             y_train_window_c_bool.append(np.array([1 if pwave_index>wind_ix[0] and pwave_index<wind_ix[1] else 0 for wind_ix in rolling_window_index_c[a_ix[count],b_ix[count]]]))
    y_train_window_c_bool = np.array(y_train_window_c_bool)
    return y_train_window_c_bool

def get_windowA(rolling_window_index_a,total_raw_data):
    window_A = []
    for tr in total_raw_data:
        ts_win_a = []
        for win_ix in rolling_window_index_a:
            original = tr.data[win_ix[0]:win_ix[1]]
            freq_domain = dct(original)
            original_scaled = tr.normalize().data[win_ix[0]:win_ix[1]]
            freq_scaled = (freq_domain - np.mean(freq_domain, axis=0)) / np.std(freq_domain, axis=0)
            ts_win_a.append(np.concatenate([original_scaled,dct(freq_scaled)]))
        window_A.append(ts_win_a)
    window_A = np.array(window_A)
    window_A = window_A.reshape(-1,304,90,2)
    return window_A

def get_windowB(ts_ix,chosen_index_a,pred_label,X_train,rolling_window_index_b,window_A):
    window_B = []
    a_ix = chosen_index_a[pred_label==1] # out of 225 windows
    for ix, tr in enumerate(X_train[pred_label==1]):
        ts_win_b = []
        chosen_wind_a = window_A[ts_ix[ix],a_ix[ix],:,0] # take only the time domain
        for win_ix in rolling_window_index_b[0,:,:]:
            original = chosen_wind_a[win_ix[0]:win_ix[1]]
            freq_domain = dct(original)
            original_scaled = (original - np.mean(original, axis=0)) / np.std(original, axis=0)
            freq_scaled = (freq_domain - np.mean(freq_domain, axis=0)) / np.std(freq_domain, axis=0)
            ts_win_b.append(np.concatenate([original_scaled,dct(freq_scaled)]))  
        window_B.append(ts_win_b)
    window_B = np.array(window_B)
    window_B = window_B.reshape(-1,15,45,2)
    return (window_B,a_ix)

def get_windowC(b_ix,window_B,rolling_window_index_c):
    window_C = []
    for ix, row in enumerate(window_B):
        ts_win_c = []
        chosen_wind_b = row[b_ix[ix],:,0]
        for win_ix in rolling_window_index_c[0,0,:,:]: # Check this
            original = chosen_wind_b[win_ix[0]:win_ix[1]]
            freq_domain = dct(original)
            original_scaled = (original - np.mean(original, axis=0)) / np.std(original, axis=0)
            freq_scaled = (freq_domain - np.mean(freq_domain, axis=0)) / np.std(freq_domain, axis=0)
            ts_win_c.append(np.concatenate([original_scaled,dct(freq_scaled)]))  
        window_C.append(ts_win_c)
    window_C = np.array(window_C)
    window_C = window_C.reshape(-1,7,25,2)
    return window_C   

def final_pred(pred_label,a_ix,b_ix,c_ix,rolling_window_index_c,total_raw_data,chosen_ix_train):
    prediction = []
    ticker = 0
    for ix, x in enumerate(pred_label):
        if x==1:
          prediction.append(rolling_window_index_c[a_ix[ticker,],b_ix[ticker,],c_ix[ticker,],0])
          ticker += 1
        else:
          prediction.append(None)

    for ix , idx in enumerate(chosen_ix_train):
        starttime = total_raw_data[idx].stats.starttime #
        freq = total_raw_data[idx].stats.sampling_rate
        if prediction[ix] is None:
           continue
        else:
            prediction[ix] = starttime + prediction[ix]/freq

    prediction = np.array(prediction)
    return prediction

