#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import load_model
import keras as ks
from keras.layers import Input, Dense,  LSTM, Flatten
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling1D, Flatten, Dropout, Dense,Conv1D



def modelA(X_train,y_train):
    model = Sequential()
    model.add(Conv2D(20,kernel_size = (20,2),activation = 'relu',padding = 'same', input_shape = (304,90,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(10,kernel_size = (30,1),activation = 'relu',padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(304,activation = 'sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer='adam',metrics=["accuracy"])
    callback = EarlyStopping(monitor='loss', patience=5)
    model.fit(X_train,y_train,epochs = 10, shuffle = True, callbacks = [callback],validation_split = 0.1)
    model.save("modelA.h5")
    

def modelB(X_train,y_train):
 
    model = Sequential()
    model.add(Conv2D(20,kernel_size = (5,2),activation = 'relu',padding = 'same', input_shape = (15,45,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(10,kernel_size = (10,1),activation = 'relu',padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(15,activation = 'sigmoid'))  # Add number of windows
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer='adam',metrics=["accuracy"])
    callback = EarlyStopping(monitor='loss', patience=5)
    model.fit(X_train,y_train,epochs = 10, shuffle = True, callbacks = [callback],validation_split = 0.1)
    model.save("modelB.h5")

def model_picker(X_train,y_train):
    model = Sequential()
    model.add(Conv2D(20,kernel_size = (3,2),activation = 'relu',padding = 'same', input_shape = (7,25,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(10,kernel_size = (5,1),activation = 'relu',padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7,activation = 'sigmoid'))  # Add 
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer='adam',metrics=["accuracy"])
    callback = EarlyStopping(monitor='loss', patience=5)
    model.fit(X_train,y_train,epochs = 10, shuffle = True, callbacks = [callback],validation_split = 0.1)
    model.save("model_picker.h5")
    


def windowA_pred(total_raw_data):

    model = tf.keras.models.load_model('modelA.h5')
    pred = model.predict(total_raw_data)
    threshold = 0.7 # after smote will work better
    pred_label = []
    chosen_index_a = []
    ts_ix = []
    for ix,pred_wind in enumerate(pred):
        if max(pred_wind)>threshold: # for futrue improvement, get the total vote count for pwave
            pred_label.append(1)
            chosen_index_a.append(np.where(pred_wind == np.amax(pred_wind))[0][0])
            ts_ix.append(ix)
        else:
            pred_label.append(0)
            chosen_index_a.append(None)
    pred_label = np.array(pred_label)
    chosen_index_a = np.array(chosen_index_a)
    ts_ix = np.array(ts_ix)
    return (ts_ix,chosen_index_a,pred_label)

def windowB_pred(X_test):

    model = tf.keras.models.load_model('modelB.h5')  
    pred_b = model.predict(X_test)
    b_ix = []
    for ix,pred_wind in enumerate(pred_b):
        b_ix.append(np.where(pred_wind == np.amax(pred_wind))[0][0])
    b_ix = np.array(b_ix)  
    return b_ix

def windowC_pred(X_test):
    model = tf.keras.models.load_model('model_picker.h5')
    pred_c = model.predict(X_test)
    c_ix = []
    for ix,pred_wind in enumerate(pred_c):
        c_ix.append(np.where(pred_wind == np.amax(pred_wind))[0][0])
    c_ix = np.array(c_ix)
    return c_ix

def training(total_raw_data,total_index,chosen_ix_train):
    rolling_window_index_a,rolling_window_index_b,rolling_window_index_c = rolling_window_index()
    window_A = get_windowA(rolling_window_index_a,total_raw_data)
    modelA(window_A,windowA_target(total_index,rolling_window_index_a)) # Fitting Model for window A
    ts_ix,chosen_index_a,pred_label = windowA_pred(window_A)  # need to check this
    window_B,a_ix = get_windowB(ts_ix,chosen_index_a,pred_label,window_A,rolling_window_index_b,window_A)
    print(window_B.dtype)
    del window_A
    # window_B = tf.convert_to_tensor(window_B)
    modelB(window_B,windowB_target(ts_ix,chosen_ix_train,total_index,a_ix,rolling_window_index_b)) # Need to check this
    b_ix = windowB_pred(window_B)
    window_C = get_windowC(b_ix,window_B,rolling_window_index_c)
    del window_B
    model_picker(window_C,windowC_target(ts_ix,chosen_ix_train,total_index,a_ix,b_ix,rolling_window_index_c))
    c_ix = windowC_pred(window_C) # Need to Check
    predictions = final_pred(pred_label,a_ix,b_ix,c_ix,rolling_window_index_c,total_raw_data,chosen_ix_train)
    return predictions

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

