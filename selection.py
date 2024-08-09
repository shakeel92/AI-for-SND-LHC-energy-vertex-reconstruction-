#!/usr/bin/env python
# coding: utf-8

# # ML approach for neutrino selection
# 
# This notebook demonstrates an overview of the current status of the neural network study.

# #### Importing packages (some of them are, in fact, not used)

# In[1]:


#pip install tensorflow


# In[3]:


#pip install --ignore-installed --upgrade tensorflow 


# In[5]:

from sklearn.metrics import classification_report  as creport
from sklearn.metrics import confusion_matrix as cfm
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os, re, gc, shutil, time
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import skimage.transform
# from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# from cnn_utils import *
import keras
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Concatenate, Add
#from tensorflow.keras import optimizers, metrics, losses
import tensorflow.keras.optimizers, tensorflow.keras.metrics, tensorflow.keras.losses
from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils import HDF5Matrix, to_categorical

import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects
K.set_image_data_format('channels_last')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


# #### Wall coordinates
# 
# Geometrical configurations for the wall selection and for the hit 2D visualisations (in matplotlib)

# In[6]:


wall_coords = [0, 300.+1, 313.+1, 326.+1, 339.+1, 352.+3]
# event_map_w = event_map * (event_meta[:,0]>wall_coords[i_wall]) * (event_meta[:,0]<wall_coords[i_wall+1])


# In[7]:


imextent = {
    'scifi_h':(0, 5, 15, 54),
    'scifi_v':(0, 5, -46, -7),
    'us_h':(0, 5, 12, 66),
    'us_v':(0, 5, 4, -79),
    'ds_h':(0, 4, 7, 71),
    'ds_v':(0, 4, 2, -79),
}


# In[ ]:


#keras_model.save('path_to_save_model.h5')


# ## Data formation

# In[52]:

#pathv='/afs/cern.ch/work/s/shahmad/SND/artem/artem_Code/neutrino_ml/data_process/'
pathv='/afs/cern.ch/work/s/shahmad/SND/artem/artem_Code/out/tmp/'
### Loading data from .npy files into hdf5

with h5py.File(pathv+'data.h5', 'a') as df:
     for name in os.listdir(pathv):
         if not name.endswith(".npy"): continue
         data_tmp = np.load(pathv+name)
         if data_tmp.ndim > 2:
            data_tmp = np.transpose(data_tmp, axes=(0,2,3,1))
         df.create_dataset(name[:-4], data=data_tmp)

print("h5 created")

with h5py.File(pathv+'data.h5', 'r') as df:
    for n in df.keys():
        if n=='ids' or n=='label': continue
        #print(n, df[n].shape)
df.close()


# In[55]:


## KEYS OF METADATA
## event_meta[i_event] = (event.MCTrack[0].GetStartZ(), event.MCTrack[0].GetPdgCode(), event.MCTrack[1].GetPdgCode())
with h5py.File(pathv+'data.h5', 'r') as df:
    metadata = df['event_metadata'][...]
map_ve = (np.abs(metadata[:,1])==12) * (np.abs(metadata[:,2])!=12)
map_nc = (np.abs(metadata[:,1])==np.abs(metadata[:,2]))# * ((np.abs(metadata[:,1])!=12))
map_vm_cc = (np.abs(metadata[:,1])==14) * (np.abs(metadata[:,2])!=14)
print('meta', metadata[1:100][6:7])
#metadata = df['event_metadata'][...]
df.close()


# In[17]:


print('ve:', np.sum(map_ve))
print('nc:', np.sum(map_nc))
print('vm_cc:', np.sum(map_vm_cc))


# In[56]:

from sklearn.model_selection import train_test_split


# In[57]:


def which_wall(pos, wall_coords):
    for i in range(5):
        if (pos>wall_coords[i]) and (pos<wall_coords[i+1]):
            return i
    return -1


# In[66]:


ids_ve = np.arange(metadata.shape[0])[map_ve]
ids_nc = np.arange(metadata.shape[0])[map_nc]
ids_vm_cc = np.arange(metadata.shape[0])[map_vm_cc]

all_ids = np.concatenate((ids_ve, ids_vm_cc, ids_nc))
all_labels = np.concatenate((np.ones_like(ids_ve)*0, np.ones_like(ids_vm_cc)*1, np.ones_like(ids_nc)*2))
all_walls = np.array([which_wall(metadata[id_ev, 0], wall_coords) for id_ev in all_ids])
if np.any(all_walls<0): print(all_ids[all_walls<0], '\n', metadata[all_ids[all_walls<0], 0], '\n', metadata[all_ids[all_walls<0], 1])
    

test_size = 0.1
tr_ids, test_ids, tr_lab, test_lab = train_test_split(all_ids, all_labels, test_size=test_size, shuffle=True)
print('train:')
print(np.unique(tr_lab, return_counts=True))
print('test:')
print(np.unique(test_lab, return_counts=True))

tr_ids, val_ids, tr_lab, val_lab = train_test_split(tr_ids, tr_lab, test_size=0.111, shuffle=True)  #90*0.111=10
print('valid:')
print(np.unique(val_lab, return_counts=True))



with h5py.File(pathv+'data.h5', 'a') as df:
     df.create_dataset('ids/train', data=tr_ids)
     df.create_dataset('ids/test', data=test_ids)
     df.create_dataset('ids/val', data=val_ids)
     df.create_dataset('label/train', data=tr_lab)
     df.create_dataset('label/test', data=test_lab)
     df.create_dataset('label/val', data=val_lab)

df.close()
with h5py.File(pathv+'data.h5', 'a') as df:
     df["ids/train"][-tr_ids.shape[0]:] = tr_ids

exit()
#############################################################
with h5py.File(pathv+'data.h5', 'r') as df:
      
     print(df.keys())
     tr_ids = df['ids/train'][...]
     test_ids = df['ids/test'][...]
     val_ids=df['ids/val'][...]
     tr_lab = df['label/train'][...]
     test_lab = df['label/test'][...]
     val_lab = df['label/val'][...]


# In[ ]:





# In[68]:


tr_sort = np.argsort(tr_ids)
tr_ids = tr_ids[tr_sort].astype(int)
tr_lab = tr_lab[tr_sort]
tr_walls = np.array([which_wall(metadata[id_ev, 0], wall_coords) for id_ev in tr_ids])
if np.any(tr_walls<0):
    print(tr_ids[tr_walls<0], '\n', metadata[tr_ids[tr_walls<0], 0], '\n', metadata[tr_ids[tr_walls<0], 1])
    raise ValueError("event outside scifi walls")

test_sort = np.argsort(test_ids)
test_ids = test_ids[test_sort].astype(int)
test_lab = test_lab[test_sort]
test_walls = np.array([which_wall(metadata[id_ev, 0], wall_coords) for id_ev in test_ids])
if np.any(test_walls<0): 
    print(test_ids[test_walls<0], '\n', metadata[test_ids[test_walls<0], 0], '\n', metadata[test_ids[test_walls<0], 1])
    raise ValueError("event outside scifi walls")


# In[ ]:
val_sort = np.argsort(val_ids)
val_ids = val_ids[val_sort].astype(int)
val_lab = val_lab[val_sort]
val_walls = np.array([which_wall(metadata[id_ev, 0], wall_coords) for id_ev in val_ids])
if np.any(val_walls<0):
    print(val_ids[test_walls<0], '\n', metadata[val_ids[test_walls<0], 0], '\n', metadata[val_ids[val_walls<0], 1])
    raise ValueError("event outside scifi walls")






# In[69]:


det_names = ['scifi', 'us', 'ds']
x_tr = {}
x_test = {}
x_val={}
Energy={}
Energy['E1'] =metadata[:][5].copy() 
Energy['E2'] = metadata[:][6].copy()
print(len(tr_ids))
print(metadata[1:100][5:6])
with h5py.File(pathv+'data.h5', 'r') as df:
    for name in det_names:
        tmp_data = df[name][...]
        print(name, tmp_data[tr_ids].shape)
        if name=='scifi':
            x_tr['scifi_h'] = (tmp_data[tr_ids, :, :, 0].copy())[..., np.newaxis]
            x_tr['scifi_v'] = (tmp_data[tr_ids, :, :, 1].copy())[..., np.newaxis]

            print('tr scifi_h: ', x_tr['scifi_h'].shape)
            print('tr scifi_v: ', x_tr['scifi_v'].shape)
            x_test['scifi_h'] = (tmp_data[test_ids, :, :, 0].copy())[..., np.newaxis]
            x_test['scifi_v'] = (tmp_data[test_ids, :, :, 1].copy())[..., np.newaxis]
            x_val['scifi_h'] = (tmp_data[val_ids, :, :, 0].copy())[..., np.newaxis]
            x_val['scifi_v'] = (tmp_data[val_ids, :, :, 1].copy())[..., np.newaxis]

            print('tr scifi_h: ', x_test['scifi_h'].shape)
            print('tr scifi_v: ', x_test['scifi_v'].shape)
            # x_test[name] = tmp_data[test_ids].copy()
            # print('test '+name+': ', x_test[name].shape)
        else:
            x_tr[name] = tmp_data[tr_ids].copy()
            print('tr '+name+': ', x_tr[name].shape)
            x_test[name] = tmp_data[test_ids].copy()
            print('test '+name+': ', x_test[name].shape)
            x_val[name] = tmp_data[val_ids].copy()
            print('val '+name+': ', x_val[name].shape)
gc.collect();


# In[ ]:





# ## Toy model definition

# ### Full model

# In[71]:


n_cl = 3
n_wall = 5
scifi_shape=(5,1536,1)
us_shape=(5,10,2)
ds_shape=(4,60,2)
drop_middle_scifi = 0.5
drop_middle_us = 0.5
drop_middle_ds = 0.5

lr=2e-4 #; betaa1=30; betaa2=100; decay=1e-3

### SciFi
scifi_h_input = Input(scifi_shape, name='scifi_h_in')
X_scifi_h = Conv2D(16, kernel_size=(3,4), padding='same', activation='elu', name='scifi_h_conv1_1')(scifi_h_input)
X_scifi_h = Conv2D(16, kernel_size=(3,4), padding='same', activation='elu', name='scifi_h_conv1_2')(X_scifi_h)
X_scifi_h = MaxPooling2D(pool_size=(1,4), padding='valid', name='scifi_h_pool1')(X_scifi_h)

X_scifi_h = Conv2D(32, kernel_size=(3,4), padding='same', activation='elu', name='scifi_h_conv2_1')(X_scifi_h)
X_scifi_h = MaxPooling2D(pool_size=(1,4), padding='valid', name='scifi_h_pool2')(X_scifi_h)

X_scifi_h = Dropout(rate=drop_middle_scifi, name='scifi_h_drop_middle')(X_scifi_h)

X_scifi_h = Conv2D(32, kernel_size=(3,4), padding='same', activation='elu', name='scifi_h_conv3_1')(X_scifi_h)
X_scifi_h = MaxPooling2D(pool_size=(1,2), padding='valid', name='scifi_h_pool3')(X_scifi_h)

X_scifi_h = Conv2D(64, kernel_size=(3,3), padding='same', activation='elu', name='scifi_h_conv4_1')(X_scifi_h)
X_scifi_h = Conv2D(64, kernel_size=(3,3), padding='same', activation='elu', name='scifi_h_conv4_2')(X_scifi_h)
X_scifi_h = MaxPooling2D(pool_size=(2,2), padding='same', name='scifi_h_pool4')(X_scifi_h)
X_scifi_h = Flatten()(X_scifi_h)

### SciFi
scifi_v_input = Input(scifi_shape, name='scifi_v_in')
X_scifi_v = Conv2D(16, kernel_size=(3,4), padding='same', activation='elu', name='scifi_v_conv1_1')(scifi_v_input)
X_scifi_v = Conv2D(16, kernel_size=(3,4), padding='same', activation='elu', name='scifi_v_conv1_2')(X_scifi_v)
X_scifi_v = MaxPooling2D(pool_size=(1,4), padding='valid', name='scifi_v_pool1')(X_scifi_v)

X_scifi_v = Conv2D(32, kernel_size=(3,4), padding='same', activation='elu', name='scifi_v_conv2_1')(X_scifi_v)
X_scifi_v = MaxPooling2D(pool_size=(1,4), padding='valid', name='scifi_v_pool2')(X_scifi_v)

X_scifi_v = Dropout(rate=drop_middle_scifi, name='scifi_v_drop_middle')(X_scifi_v)

X_scifi_v = Conv2D(32, kernel_size=(3,4), padding='same', activation='elu', name='scifi_v_conv3_1')(X_scifi_v)
X_scifi_v = MaxPooling2D(pool_size=(1,2), padding='valid', name='scifi_v_pool3')(X_scifi_v)

X_scifi_v = Conv2D(64, kernel_size=(3,3), padding='same', activation='elu', name='scifi_v_conv4_1')(X_scifi_v)
X_scifi_v = Conv2D(64, kernel_size=(3,3), padding='same', activation='elu', name='scifi_v_conv4_2')(X_scifi_v)
X_scifi_v = MaxPooling2D(pool_size=(2,2), padding='same', name='scifi_v_pool4')(X_scifi_v)
X_scifi_v = Flatten()(X_scifi_v)


### US
us_input = Input(us_shape, name='us_in')
X_us = Conv2D(16, kernel_size=(3,3), padding='same', activation='elu', name='us_conv1_1')(us_input)
X_us = Conv2D(32, kernel_size=(3,3), padding='same', activation='elu', name='us_conv1_2')(X_us)
X_us = MaxPooling2D(pool_size=2, padding='same', name='us_pool1')(X_us)

X_us = Conv2D(32, kernel_size=(3,3), padding='same', activation='elu', name='us_conv2_1')(X_us)
X_us = MaxPooling2D(pool_size=2, padding='same', name='us_pool2')(X_us)

X_us = Dropout(rate=drop_middle_us, name='us_drop_middle')(X_us)

X_us = Conv2D(64, kernel_size=(3,3), padding='same', activation='elu', name='us_conv3_1')(X_us)
X_us = Flatten()(X_us)

### DS
ds_input = Input(ds_shape, name='ds_in')
X_ds = Conv2D(16, kernel_size=(3,3), padding='same', activation='elu', name='ds_conv1_1')(ds_input)
X_ds = Conv2D(32, kernel_size=(3,3), padding='same', activation='elu', name='ds_conv1_2')(X_ds)
X_ds = MaxPooling2D(pool_size=(1,2), padding='same', name='ds_pool1')(X_ds)

X_ds = Conv2D(32, kernel_size=(3,3), padding='same', activation='elu', name='ds_conv2_1')(X_ds)
X_ds = MaxPooling2D(pool_size=2, padding='same', name='ds_pool2')(X_ds)

X_ds = Dropout(rate=drop_middle_ds, name='ds_drop_middle')(X_ds)

X_ds = Conv2D(64, kernel_size=(3,3), padding='same', activation='elu', name='ds_conv3_1')(X_ds)
X_ds = Flatten()(X_ds)



X = Concatenate()([X_scifi_h, X_scifi_v, X_us, X_ds])
X_cl = Dense(n_cl, activation='softmax', name='y_class')(X)
X_wall = Dense(n_wall, activation='softmax', name='y_wall')(X)

toy_model = Model(inputs = [scifi_h_input, scifi_v_input, us_input, ds_input], outputs = [X_cl, X_wall], name='full_toy_model')

K.clear_session()

adamka = tf.keras.optimizers.Adam(learning_rate=lr)

toy_model.compile(optimizer=adamka, loss={'y_class':'categorical_crossentropy', 'y_wall':'categorical_crossentropy'}, metrics={'y_class':'categorical_accuracy', 'y_wall':'categorical_accuracy' })

'''
toy_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
'''
#toy_model.compile(optimizer=adamka, loss={'y_class':'categorical_crossentropy', 'y_wall':'categorical_crossentropy'}, metrics=['accuracy'])


# In[72]:


toy_model.summary()


# In[ ]:

xv= [x_val['scifi_h'], x_val['scifi_v'], x_val['us'], x_val['ds']]
yv=[to_categorical(val_lab, num_classes=n_cl), to_categorical(val_walls, num_classes=n_wall)]

xtr=[x_tr['scifi_h'], x_tr['scifi_v'], x_tr['us'], x_tr['ds']]
ytr=[to_categorical(tr_lab, num_classes=n_cl), to_categorical(tr_walls, num_classes=n_wall)]


class Metrics(keras.callbacks.Callback):
    
    def __init__(self, val_data=(xv, yv),tr_data=(xtr,ytr)):
        self.validation_data = val_data
        self.train_data=tr_data
    def on_train_begin(self, logs={}):
        self._data = []



    def on_epoch_end(self, batch, logs={}):
        x_test, y_test = self.validation_data[0], self.validation_data[1]
        x_train, y_train=self.train_data[0], self.train_data[1]
        #y_predict = np.asarray(toy_model.predict(x_test))

        ytr_predict=toy_model.predict(x_train)
        y_predict =toy_model.predict(x_test)

        tr_pred=np.argmax(ytr_predict[0], axis=1)
        tr_true =np.argmax(y_train[0], axis=1) 

        pred = np.argmax(y_predict[0], axis=1)       
        true =np.argmax(y_test[0], axis=1)

        #cm = confusion_matrix(true, pred)
        report=creport(true,pred)
        print(report)
        
        cm=cfm(true,pred)
        #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
        report1=creport(tr_true,tr_pred)
        cm1=cfm(tr_true,tr_pred)
        print(cm1)
        #cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
        



        self._data.append({
            'classLevelaccuracy':cm ,'report':report,  'classLevelaccuracytr':cm1 ,'reporttr':report1,
        })
        return

    def get_data(self):
        return self._data

metrics = Metrics()


# ## Train model

# In[73]:
#xv= [x_val['scifi_h'], x_val['scifi_v'], x_val['us'], x_val['ds']]
#yv=[to_categorical(val_lab, num_classes=n_cl), to_categorical(val_walls, num_classes=n_wall)]
hist = toy_model.fit(x=[x_tr['scifi_h'], x_tr['scifi_v'], x_tr['us'], x_tr['ds']], y=[to_categorical(tr_lab, num_classes=n_cl), to_categorical(tr_walls, num_classes=n_wall)], batch_size=128, epochs=25,validation_data=(xv,yv), callbacks=[metrics]) 
                                   
rpdata=metrics.get_data()
#print(rpdata.keys())                                 
#print('callbacks ',rpdata)

# In[15]:


#toy_model.save('/afs/cern.ch/work/s/shahmad/SND/artem/condor/'+'toy_neutrino_25')
PATHF='/afs/cern.ch/work/s/shahmad/SND/artem/artem_Code/out/figure/'

toy_model.save(PATHF+'toy_neutrino_e25')

with open(PATHF+'ddata', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)

with open(PATHF+'rdata', 'wb') as file_pi:
    pickle.dump(rpdata, file_pi)



'''

with open(PATHF+'trainHistoryDict', "rb") as file_pi:
    hist = pickle.load(file_pi)
'''
'''
history_df = pd.DataFrame(hist.history)
print(hist.history.keys())
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history_df['loss'], label='training loss')
plt.plot(history_df['val_loss'], label='validation loss')
plt.title('Model Loss Function')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history_df['accuracy'], label='training accuracy')
plt.plot(history_df['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.legend();
plt.savefig(PATHF+'loss'+'.jpg')
plt.close()
'''


# In[16]:


#retoy_model = tf.keras.models.load_model("toy_neutrino_25")


# In[ ]:





# In[17]:


test_pred = toy_model.predict(x=[x_test['scifi_h'], x_test['scifi_v'], x_test['us'], x_test['ds']])


# In[18]:


np.savetxt(PATHF+'test_pred_class.txt', test_pred[0], delimiter=', ', header='predictions: ve, vm, nc')
np.savetxt(PATHF+'test_pred_walls.txt', test_pred[1], delimiter=', ', header='predictions: walls 0-4')


# ## Confusion matrix

# In[19]:


## Total in row = 100%
def confusion_matrix(y_true, y_pred):
    n_class = len(np.unique(y_true))
    conf_mat = np.zeros((n_class, n_class))
    n_tot = len(y_true)
    for i in range(n_class):
        for j in range(n_class):
            conf_mat[i, j] = np.sum((y_true==i)*(y_pred==j))/np.sum((y_true==i))*100
    return np.around(conf_mat, decimals=1)


# * sum over every row is 100%

# In[20]:


fd=confusion_matrix(test_lab, np.argmax(test_pred[0], axis=-1))
print(fd)

# In[21]:


## Total in column = 100%
def confusion_purity_matrix(y_true, y_pred):
    n_class = len(np.unique(y_true))
    conf_mat = np.zeros((n_class, n_class))
    n_tot = len(y_true)
    for i in range(n_class):
        for j in range(n_class):
            conf_mat[i, j] = np.sum((y_true==i)*(y_pred==j))/np.sum((y_pred==j))*100
    return np.around(conf_mat, decimals=1)


# * sum over every column is 100%

# In[22]:


ddd=confusion_purity_matrix(test_lab, np.argmax(test_pred[0], axis=-1))
print(ddd)

# In[ ]:





# ## Study events

# In[23]:


nchan = {'scifi':1536, 'us':10, 'ds':60}
nplane = {'scifi':5, 'us':5, 'ds':4}


# #### CORRECT PREDICTIONS

# In[24]:


## CORRECT PREDICTIONS
y_test_pred = np.argmax(test_pred[0], axis=-1)
y_test_wall = np.argmax(test_pred[1], axis=-1)

class_names = ['ve', 'vm', 'nc']
hit_wall = {}
for det_name in ['scifi_h', 'scifi_v', 'us', 'ds']:
    hit_wall[det_name] = {}
    for i_wall in range(5):
        hit_wall[det_name][i_wall] = {}
        for i_cl, cl_name in enumerate(class_names):
            hit_mask = ((test_lab==i_cl) * (y_test_pred==i_cl) * (test_walls==i_wall)).astype(bool)            
            hit_wall[det_name][i_wall][cl_name] = x_test[det_name][hit_mask]

#PATHF='/afs/cern.ch/work/s/shahmad/SND/artem/figures/'

# ### Event examples

# In[25]:


# n = np.random.randint(1000)
n = 19
print('n=',n)
fig, ax = plt.subplots(1, 3, figsize=(4,5))

hit_scifi = np.zeros((nplane['scifi'], 64, 2), dtype=int)
#print(hit_wall['scifi_h'][0]['ve'][n,...,0])
tmp_scifi = hit_wall['scifi_h'][0]['ve'][n,...,0]
for i in range(64): #12
    hit_scifi[:, i, 0] = tmp_scifi[:, 24*i:24*(i+1)].sum(axis=1) #128
tmp_scifi = hit_wall['scifi_v'][0]['ve'][n,...,0]
for i in range(64):
    hit_scifi[:, i, 1] = tmp_scifi[:, 24*i:24*(i+1)].sum(axis=1)
plt.imshow(hit_scifi[...,0].T)
for i, name in enumerate(['scifi_h', 'scifi_v', 'us']):
    if 'scifi' in name:
        ax[i].imshow(hit_scifi[...,i].T, origin='lower', extent=imextent[name])
    else:
        ax[i].imshow(hit_wall['us'][0]['ve'][n,...,0].T, extent=imextent['us_h'])
    ax[i].set_title(name)
    # cbar = fig.colorbar(im, ax=ax[i])
#plt.show()
plt.savefig(PATHF+'image1.jpg')
# In[26]:

plt.close()

# n = np.random.randint(1000)
#n = 145
n=45
print('n=',n)
fig, ax = plt.subplots(1, 3, figsize=(4,5))

hit_scifi = np.zeros((nplane['scifi'], 64, 2), dtype=int)
tmp_scifi = hit_wall['scifi_h'][0]['ve'][n,...,0]
for i in range(64): #12
    hit_scifi[:, i, 0] = tmp_scifi[:, 24*i:24*(i+1)].sum(axis=1) #128
tmp_scifi = hit_wall['scifi_v'][0]['ve'][n,...,0]
for i in range(64):
    hit_scifi[:, i, 1] = tmp_scifi[:, 24*i:24*(i+1)].sum(axis=1)
plt.imshow(hit_scifi[...,0].T)

for i, name in enumerate(['scifi_h', 'scifi_v', 'us']):
    if 'scifi' in name:
        ax[i].imshow(hit_scifi[...,i].T, origin='lower', extent=imextent[name])
    else:
        ax[i].imshow(hit_wall['us'][0]['ve'][n,...,0].T, extent=imextent['us_h'])
    ax[i].set_title(name)
    # cbar = fig.colorbar(im, ax=ax[i])
#plt.show()
plt.savefig(PATHF+'image2.jpg')
plt.close()
# In[27]:


# n = np.random.randint(1000)
n = 78
print('n=',n)
fig, ax = plt.subplots(1, 3, figsize=(4,5))

hit_scifi = np.zeros((nplane['scifi'], 64, 2), dtype=int)
tmp_scifi = hit_wall['scifi_h'][0]['ve'][n,...,0]
for i in range(64): #12
    hit_scifi[:, i, 0] = tmp_scifi[:, 24*i:24*(i+1)].sum(axis=1) #128
tmp_scifi = hit_wall['scifi_v'][0]['ve'][n,...,0]
for i in range(64):
    hit_scifi[:, i, 1] = tmp_scifi[:, 24*i:24*(i+1)].sum(axis=1)
plt.imshow(hit_scifi[...,0].T)
for i, name in enumerate(['scifi_h', 'scifi_v', 'us']):
    if 'scifi' in name:
        ax[i].imshow(hit_scifi[...,i].T, origin='lower', extent=imextent[name])
    else:
        ax[i].imshow(hit_wall['us'][0]['ve'][n,...,0].T, extent=imextent['us_h'])
    ax[i].set_title(name)
    # cbar = fig.colorbar(im, ax=ax[i])
#plt.show()
plt.savefig(PATHF+'fig3.jpg')

plt.close()
# ### Visualise aggregated distributions

# In[28]:


def plot_5walls(hit_wall, class_names=None, name='scifi'):
    ### Structure: hitwall[det_name][wall][class]
    ## event_map_w = event_map * (event_meta[:,0]>wall_coords[i_wall]) * (event_meta[:,0]<wall_coords[i_wall+1])
    if class_names is None: class_names = ['ve', 'vm', 'nc']
    for i_wall in range(nplane['scifi']):
        
        i_ax = 0
        hit_tot = {}
        for i_cl, cl_name in enumerate(class_names):
            if name=='scifi':
                tmp_scifi = np.zeros((nplane['scifi'], 64, 2))
                hit_tot[cl_name] = np.sum(hit_wall['scifi_h'][i_wall][cl_name], axis=0, dtype=int)
                for i in range(64):
                    tmp_scifi[:, i, 0] = hit_tot[cl_name][:, 24*i:24*(i+1), 0].sum(axis=1)
                hit_tot[cl_name] = np.sum(hit_wall['scifi_v'][i_wall][cl_name], axis=0, dtype=int)
                for i in range(64):
                    tmp_scifi[:, i, 1] = hit_tot[cl_name][:, 24*i:24*(i+1), 0].sum(axis=1)
                hit_tot[cl_name] = tmp_scifi
            else:
                hit_tot[cl_name] = np.sum(hit_wall[name][i_wall][cl_name], axis=0, dtype=int)
        fig, ax = plt.subplots(1, 6 if name!='us' else 3, figsize=(8 if name!='us' else 4,5))
        for vert in range(2):
            if name=='us' and vert: continue
            for i_cl, cl_name in enumerate(class_names):
                im = ax[i_cl+vert*3].imshow(hit_tot[cl_name][...,vert].T, origin='lower', extent=imextent[name+'_'+('v' if vert else 'h')])
                ax[i_cl+vert*3].set_title(cl_name+' '+('vert' if vert else 'horiz'))
                # cbar = fig.colorbar(im, ax=ax[i_cl+vert*3])
                i_ax += 1
        plt.suptitle(str(i_wall)+'wall '+name)
        #plt.show()
        plt.savefig(PATHF+'image4.jpg')
        plt.close()
# In[29]:


plot_5walls(hit_wall, class_names, name='scifi')


# ### Event distributions

# In[30]:


from scipy.stats import norm
from scipy.optimize import curve_fit


# In[31]:


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def fit_gaus1d(hitmap, i_wall=0, name=None):
    hits1d = hitmap[i_wall:i_wall+2].sum(axis=0)
    mu, std = norm.fit(hits1d)
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    if name is None: p0 = [30., 30., 10.]
    else:
        p0 = [30., 10., 10.] if name.endswith('h') else [30., 50., 10.]
    coeff, var_matrix = curve_fit(gauss, np.arange(len(hits1d)), hits1d, p0=p0, bounds=(0, np.inf))
    return coeff    


# In[32]:


mu_wall = {nu_name:{} for nu_name in ['ve', 'vm', 'nc']}
std_wall = {nu_name:{} for nu_name in ['ve', 'vm', 'nc']}


# In[33]:


nu_name = 've'
div_events = ''
for name in ['scifi_h', 'scifi_v']:
    mu_wall[nu_name][name] = [[] for i in range(n_wall)]
    std_wall[nu_name][name] = [[] for i in range(n_wall)]
    for i_wall in range(n_wall):
        n_scifi = hit_wall[name][i_wall][nu_name].shape[0]
        hit_scifi = np.zeros((n_scifi, nplane['scifi'], 64), dtype=int)
        tmp_scifi = hit_wall[name][i_wall][nu_name][..., 0]
        for i in range(64): #12
            hit_scifi[:, :, i] = tmp_scifi[:, :, 24*i:24*(i+1)].sum(axis=2) #128
        
        for i_ev, hit_ev in tqdm(enumerate(hit_scifi)):
            try:
                coeff = fit_gaus1d(hit_ev, i_wall, name)
                mu_wall[nu_name][name][i_wall].append(coeff[1])
                std_wall[nu_name][name][i_wall].append(coeff[2])
                if coeff[2]<0:
                    print('neg std. ', i_ev, ' mean:', np.around(coeff[1]), ' std:', np.around(coeff[2]))
            except:
                div_events+= name+', '+str(i_wall)+'wall, '+str(i_ev)+'\n'
    for i_wall in range(n_wall):
        plt.hist(mu_wall[nu_name][name][i_wall], bins=20, range=(0,100), label='wall '+str(i_wall), alpha=0.8, density=True, histtype='step', linewidth=2)
    plt.title('mu '+name)
    plt.legend()
    #plt.show()
    plt.savefig(PATHF+'image5.jpg')
    plt.close()
    for i_wall in range(n_wall):
        plt.hist(std_wall[nu_name][name][i_wall], bins=12, range=(0,24), label='wall '+str(i_wall), alpha=0.8, density=True, histtype='step', linewidth=2)
    plt.title('std '+name)
    plt.legend()
    #plt.show()
    plt.savefig(PATHF+'image41.jpg')
print('div_n:',len(div_events.split('\n'))-1)

plt.close()
# In[34]:

nu_name = 've'
div_events = ''
for name in ['scifi_h', 'scifi_v']:
    mu_wall[nu_name][name] = [[] for i in range(n_wall)]
    std_wall[nu_name][name] = [[] for i in range(n_wall)]
    for i_wall in range(n_wall):
        n_scifi = hit_wall[name][i_wall][nu_name].shape[0]
        hit_scifi = np.zeros((n_scifi, nplane['scifi'], 64), dtype=int)
        tmp_scifi = hit_wall[name][i_wall][nu_name][..., 0]
        for i in range(64): #12
            hit_scifi[:, :, i] = tmp_scifi[:, :, 24*i:24*(i+1)].sum(axis=2) #128
        
        for i_ev, hit_ev in tqdm(enumerate(hit_scifi)):
            try:
                coeff = fit_gaus1d(hit_ev, i_wall, name)
                mu_wall[nu_name][name][i_wall].append(coeff[1])
                std_wall[nu_name][name][i_wall].append(coeff[2])
                if coeff[2]<0:
                    print('neg std. ', i_ev, ' mean:', np.around(coeff[1]), ' std:', np.around(coeff[2]))
            except:
                div_events+= name+', '+str(i_wall)+'wall, '+str(i_ev)+'\n'
    for i_wall in range(n_wall):
        plt.hist(mu_wall[nu_name][name][i_wall], bins=25 if 'h' in name else 32, range=(0,50 if 'h' in name else 64), label='wall '+str(i_wall), alpha=0.8, density=True, histtype='step', linewidth=2)
    plt.title('mu '+name)
    plt.legend()
    plt.minorticks_on()
    #plt.show()
    plt.savefig(PATHF+'image44.jpg')
    plt.close()
    for i_wall in range(n_wall):
        plt.hist(std_wall[nu_name][name][i_wall], bins=12, range=(0,24), label='wall '+str(i_wall), alpha=0.8, density=True, histtype='step', linewidth=2)
    plt.title('std '+name)
    plt.legend()
    #plt.show()
    plt.savefig(PATHF+'image5.jpg')
    plt.close()
print('div_n:',len(div_events.split('\n'))-1)


nu_name = 'vm'
div_events = ''
for name in ['scifi_h', 'scifi_v']:
    mu_wall[nu_name][name] = [[] for i in range(n_wall)]
    std_wall[nu_name][name] = [[] for i in range(n_wall)]
    for i_wall in range(n_wall):
        n_scifi = hit_wall[name][i_wall][nu_name].shape[0]
        hit_scifi = np.zeros((n_scifi, nplane['scifi'], 64), dtype=int)
        tmp_scifi = hit_wall[name][i_wall][nu_name][..., 0]
        for i in range(64): #12
            hit_scifi[:, :, i] = tmp_scifi[:, :, 24*i:24*(i+1)].sum(axis=2) #128
        
        for i_ev, hit_ev in tqdm(enumerate(hit_scifi)):
            try:
                coeff = fit_gaus1d(hit_ev, i_wall, name)
                mu_wall[nu_name][name][i_wall].append(coeff[1])
                std_wall[nu_name][name][i_wall].append(coeff[2])
                if coeff[2]<0:
                    print('neg std. ', i_ev, ' mean:', np.around(coeff[1]), ' std:', np.around(coeff[2]))
            except:
                div_events+= name+', '+str(i_wall)+'wall, '+str(i_ev)+'\n'
    for i_wall in range(n_wall):
        plt.hist(mu_wall[nu_name][name][i_wall], bins=20, range=(0,100), label='wall '+str(i_wall), alpha=0.8, density=True, histtype='step', linewidth=2)
    plt.title('mu '+name)
    plt.legend()
    plt.savefig(PATHF+'image61.jpg')
    #plt.show()
    for i_wall in range(n_wall):
        plt.hist(std_wall[nu_name][name][i_wall], bins=12, range=(0,24), label='wall '+str(i_wall), alpha=0.8, density=True, histtype='step', linewidth=2)
    plt.title('std '+name)
    plt.legend()
    #plt.show()
    plt.savefig(PATHF+'image7.jpg')
    plt.close()
print('div_n:',len(div_events.split('\n'))-1)


# In[36]:


nu_name = 'nc'
div_events = ''
for name in ['scifi_h', 'scifi_v']:
    mu_wall[nu_name][name] = [[] for i in range(n_wall)]
    std_wall[nu_name][name] = [[] for i in range(n_wall)]
    for i_wall in range(n_wall):
        n_scifi = hit_wall[name][i_wall][nu_name].shape[0]
        hit_scifi = np.zeros((n_scifi, nplane['scifi'], 64), dtype=int)
        tmp_scifi = hit_wall[name][i_wall][nu_name][..., 0]
        for i in range(64): #12
            hit_scifi[:, :, i] = tmp_scifi[:, :, 24*i:24*(i+1)].sum(axis=2) #128
        
        for i_ev, hit_ev in tqdm(enumerate(hit_scifi)):
            try:
                coeff = fit_gaus1d(hit_ev, i_wall, name)
                mu_wall[nu_name][name][i_wall].append(coeff[1])
                std_wall[nu_name][name][i_wall].append(coeff[2])
                if coeff[2]<0:
                    print('neg std. ', i_ev, ' mean:', np.around(coeff[1]), ' std:', np.around(coeff[2]))
            except:
                div_events+= name+', '+str(i_wall)+'wall, '+str(i_ev)+'\n'
    for i_wall in range(n_wall):
        plt.hist(mu_wall[nu_name][name][i_wall], bins=20, range=(0,100), label='wall '+str(i_wall), alpha=0.8, density=True, histtype='step', linewidth=2)
    plt.title('mu '+name)
    plt.legend()
    #plt.show()
    plt.savefig(PATHF+'image8.jpg')
    plt.close()
    for i_wall in range(n_wall):
        plt.hist(std_wall[nu_name][name][i_wall], bins=12, range=(0,24), label='wall '+str(i_wall), alpha=0.8, density=True, histtype='step', linewidth=2)
    plt.title('std '+name)
    plt.legend()
    #plt.show()
    plt.savefig(PATHF+'image81.jpg')
    plt.close()
print('div_n:',len(div_events.split('\n'))-1)


# ### STD filter separator

# In[37]:


# std_thresh = 4
std_thr_range = np.arange(1,10, 0.5)
thr_perc = {nu_name:{name:[] for name in ['scifi_h', 'scifi_v']} for nu_name in ['ve', 'vm', 'nc']}
for std_thresh in std_thr_range:
    #print('thr:', std_thresh, end='\t')
    for nu_name in ['ve', 'vm', 'nc']:
        n_tot, n_thr = 0, 0
        for name in ['scifi_h', 'scifi_v']:
            n_tot_name, n_thr_name = 0, 0
            for i_wall in range(5):
                n_tot_name += len(std_wall[nu_name][name][i_wall])
                n_thr_name += np.sum(np.array(std_wall[nu_name][name][i_wall])>std_thresh)
            n_tot += n_tot_name
            n_thr += n_thr_name
            thr_perc[nu_name][name].append(np.around(n_thr_name/n_tot_name*100, decimals=1))
        #print(nu_name, ':', np.around(n_thr/n_tot*100, decimals=1), '%', end='\t')
    #print()

plt.plot(std_thr_range, thr_perc['ve']['scifi_h'], 'blue', label='ve '+'scifi_h')
plt.plot(std_thr_range, thr_perc['vm']['scifi_h'], 'orange', label='vm '+'scifi_h')
plt.plot(std_thr_range, thr_perc['nc']['scifi_h'], 'green', label='nc '+'scifi_h')

plt.plot(std_thr_range, thr_perc['ve']['scifi_v'], 'blue', ls='dashed', label='ve '+'scifi_v')
plt.plot(std_thr_range, thr_perc['vm']['scifi_v'], 'orange', ls='dashed', label='vm '+'scifi_v')
plt.plot(std_thr_range, thr_perc['nc']['scifi_v'], 'green', ls='dashed', label='nc '+'scifi_v')
plt.title('scifi')
plt.grid()
plt.legend()
#plt.show()
plt.savefig(PATHF+'ff.jpg')

# ### Confusion matrix with variable threshold for Ve

# * we try to enforce higher purity for Ve by requiring minimal threshold on algorithm's "confidence"

# In[38]:
plt.close()


def ve_thresh(y_pred, thresh=0.7):
    ve_mask = (y_pred[:,0]>thresh) * (np.argmax(y_pred, axis=-1)==0)
    nonve_mask = np.invert(ve_mask)
    preds = np.zeros(y_pred.shape[0])
    preds[nonve_mask] = np.argmax(y_pred[nonve_mask,1:], axis=-1)+1
    return preds


# * confusion matrix

# In[39]:


confusion_matrix(test_lab, ve_thresh(test_pred[0], thresh=0.7))


# In[40]:


confusion_matrix(test_lab, ve_thresh(test_pred[0], thresh=0.8))


# In[41]:


confusion_matrix(test_lab, ve_thresh(test_pred[0], thresh=0.87))


# * purity matrix

# In[42]:


confusion_purity_matrix(test_lab, ve_thresh(test_pred[0], thresh=0.7))


# In[43]:


confusion_purity_matrix(test_lab, ve_thresh(test_pred[0], thresh=0.8))


# In[44]:


confusion_purity_matrix(test_lab, ve_thresh(test_pred[0], thresh=0.87))


# In[ ]:





# # Expected gradients - relevance map visualisation

# The relevance map is obtained by
# * Integrated Gradients: https://arxiv.org/pdf/1703.01365.pdf
# 
# with the extension aimed to reduce noise and make the map more robust (by averaging the Gaussian noise baseline)
# * Expected Gradients: https://arxiv.org/pdf/1906.10670.pdf

# Great demonstration and explanation of several methods for relevance map visuallisations:
# * https://distill.pub/2020/attribution-baselines/

# In[ ]:





# *KEY BASELINES* that can be relevant to compare (to see which one produces more physically legitimate pattern)
# * **INPUT BASELINE** (average over all, or some, inputs to achieve a correct physical distribution in the baseline)
# * **INPUT+NOISE BASELINE** (same, but with gaussian noise added to each baseline, to increase randomness and reduce artefacts)
# * **ZERO BASELINE** (baseline of array with all 0)
# * **ZERO+NOISE BASELINE** (simply gaussian noise in each pixel as a baseline, with some physically motivated std, average on multiple runs)

# In[45]:


def get_gradients(img_input, top_pred_idx, toy_model):#, top_pred_idx):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    img_scifi_h = tf.cast(img_input[0], tf.float32)
    img_scifi_v = tf.cast(img_input[1], tf.float32)
    img_us = tf.cast(img_input[2], tf.float32)
    img_ds = tf.cast(img_input[3], tf.float32)
    images_list = [img_scifi_h, img_scifi_v, img_us, img_ds]

    with tf.GradientTape() as tape:
        tape.watch(img_scifi_h)
        tape.watch(img_scifi_v)
        tape.watch(img_us)
        tape.watch(img_ds)
        preds = toy_model(images_list)
        top_class = preds[0][:, top_pred_idx] # [0] to take class prediction, not walls
    return tape.gradient(top_class, images_list)


# In[46]:


### Currently not used
### Implementation that can calculate feature maps relative to the input with gaussian noise.
def expected_gauss_gradients(img_input, toy_model, top_pred_idx, num_samples=100, noise=None, baseline='zero'):
    """Computes Integrated Gradients for a predicted label.
        Expected Gradients paper: https://arxiv.org/pdf/1906.10670.pdf
        Integrated Gradients initial paper: https://arxiv.org/pdf/1703.01365.pdf
    Args:
        img_input (ndarray): Original image
        top_pred_idx: Predicted label for the input image
        toy_model: Keras model to obtain gradients from
        baseline_imgs (ndarray): The set of baseline images to sample from. Should follow the distribution of the original image.
        num_samples: Number of sampling steps between to estimate the integral between baseline and input, and the integral over distribution. These
            steps determine the estimation error.
        noise: std of the gaussian noise.
        baseline (str): 'zero' or 'input' defining whether noise is used alone or added on top of the input.

    Returns:
        Expected gradients w.r.t input image and a set of baseline images
    """
    if noise is None: noise = [0.1, 0.1, 0.1, 0.1]
    grad_input, base_samples = [], []
    for i, name in enumerate(['scifi_h', 'scifi_v', 'us', 'ds']):
        img_size = img_input[i].shape
        img_input[i] = img_input[i][np.newaxis,...]
        base_samples.append(np.random.randn(num_samples,*img_size)*noise[i] + (0 if baseline=='zero' else img_input[i]))
        alpha_samples = np.random.random_sample(size=(num_samples,1,1,1))
        grad_input.append(base_samples[i] + alpha_samples*(img_input[i] - base_samples[i]))
    
    grads = get_gradients(grad_input, top_pred_idx, toy_model) # tf.convert_to_tensor(get_gradients(grad_input, top_pred_idx, toy_model), dtype=tf.float32)
    
    grads = [tf.reduce_mean(tf.convert_to_tensor((img_input[i] - base_samples[i])*grads[i]), axis=0) for i in range(4)]
    return grads # tf.reduce_mean(grads, axis=0)


# In[ ]:





# ## Zero-base gradients (no noise)

# In[47]:


def zero_integrated_gradients(img_input, toy_model, top_pred_idx, num_samples=100):
    """Computes Integrated Gradients for a predicted label.
        Expected Gradients paper: https://arxiv.org/pdf/1906.10670.pdf
        Integrated Gradients initial paper: https://arxiv.org/pdf/1703.01365.pdf
    Args:
        img_input (ndarray): Original image
        top_pred_idx: Predicted label for the input image
        toy_model: Keras model to obtain gradients from
        baseline_imgs (ndarray): The set of baseline images to sample from. Should follow the distribution of the original image.
        num_samples: Number of sampling steps between to estimate the integral between baseline and input, and the integral over distribution. These
            steps determine the estimation error.
        noise: std of the gaussian noise.
        baseline (str): 'zero' or 'input' defining whether noise is used alone or added on top of the input.

    Returns:
        Expected gradients w.r.t input image and a set of baseline images
    """
    grad_input = [np.zeros((1, *img_input[i].shape)) for i in range(4)]
    
    grads = get_gradients(grad_input, top_pred_idx, toy_model) # tf.convert_to_tensor(get_gradients(grad_input, top_pred_idx, toy_model), dtype=tf.float32)
    
    grads = [tf.convert_to_tensor(grads[i]) for i in range(4)]
    return grads # tf.reduce_mean(grads, axis=0)


# In[48]:


def test_sample_zero_gradient_walls(X_eg, y_eg, wall_ids, toy_model, target_nu=0, num_samples=50, return_std=False):
    """
    Average the expected gradients over test samples for a selected class
    """
    n_walls = np.max(wall_ids)+1
    wall_ids = wall_ids[y_eg==target_nu]
    X_nu = [X[y_eg==target_nu] for X in X_eg] # X_eg[y_eg==target_nu] # 
    eg_nu = {i_wall: [[], [], [], []] for i_wall in range(n_walls)}
    for id_eg in tqdm(range(X_nu[0].shape[0]), desc='EG average'):
        i_wall = wall_ids[id_eg]
        im_eg = [X_nu[0][id_eg], X_nu[1][id_eg], X_nu[2][id_eg], X_nu[3][id_eg]] # X_nu[id_eg] #
        # exp_grad_tmp = expected_gauss_gradients(im_eg, toy_model=toy_model, top_pred_idx=target_nu, num_samples=num_samples, noise=noise, baseline=baseline)
        exp_grad_tmp = zero_integrated_gradients(im_eg, toy_model=toy_model, top_pred_idx=target_nu, num_samples=num_samples)
        for i in range(4):
            eg_nu[i_wall][i].append(exp_grad_tmp[i])
    eg_nu = {i_wall:[np.array(eg_nu[i_wall][i]) for i in range(4)] for i_wall in range(n_walls)}
    if return_std:
        return {i_wall:[np.squeeze(np.mean(eg_nu[i_wall][i], axis=0)) for i in range(4)] for i_wall in range(n_walls)}, {i_wall:[np.squeeze(np.std(eg_nu[i_wall][i], axis=0)) for i in range(4)] for i_wall in range(n_walls)}
    return {i_wall:[np.squeeze(np.mean(eg_nu[i_wall][i], axis=0)) for i in range(4)] for i_wall in range(n_walls)}


# ### V_e

# In[50]:


X_test = [x_test['scifi_h'], x_test['scifi_v'], x_test['us'], x_test['ds']]


# In[51]:


target_nu = 0

# Estimating the pixel variance from the test data in the corresponding period
#sigma_adapt = [np.std(X_test[i][test_lab==target_nu], axis=0).mean() for i in range(3)]
# Running expected gradients estimation with a gaussian baseline around input data. The gaussian noise corresponds to input pixel variance.
toy_gradient_wall_output_zerozero, toy_gradient_wall_std_zerozero = test_sample_zero_gradient_walls(X_test, test_lab, test_walls, toy_model=toy_model, target_nu=target_nu, num_samples=50, return_std=True)


# In[ ]:





# In[52]:


## Ve GRAD walls
hit_wall_grad_zero = {}
for i, det_name in enumerate(['scifi_h', 'scifi_v', 'us', 'ds']):
    hit_wall_grad_zero[det_name] = {}
    for i_wall in range(5):
        hit_wall_grad_zero[det_name][i_wall] = toy_gradient_wall_output_zerozero[i_wall][i]


# In[54]:


def plot_grad_walls(hitmap, title='Ve', cmap='RdBu_r'):
    # full_min = np.min([[np.min(hit) for hit in hitw.values()] for hitw in hitmap.values()])
    # full_max = np.max([[np.max(hit) for hit in hitw.values()] for hitw in hitmap.values()])
    
    for i_wall in range(nplane['scifi']):
        fig, ax = plt.subplots(1, 3, figsize=(7,5))
        i_ax = 0
        hit_tot = {}
        # event_map_w = event_map * (event_meta[:,0]>wall_coords[i_wall]) * (event_meta[:,0]<wall_coords[i_wall+1])
        hit_tot['scifi_h'] = np.copy(hitmap['scifi_h'][i_wall]) # np.sum(hitmap['scifi'], axis=0, dtype=int)
        tmp_scifi = np.zeros((nplane['scifi'], 64))
        for i in range(64):
            tmp_scifi[:, i] = hit_tot['scifi_h'][:, 24*i:24*(i+1)].sum(axis=1)
        hit_tot['scifi_h'] = tmp_scifi

        hit_tot['scifi_v'] = np.copy(hitmap['scifi_v'][i_wall]) # np.sum(hitmap['scifi'], axis=0, dtype=int)
        tmp_scifi = np.zeros((nplane['scifi'], 64))
        for i in range(64):
            tmp_scifi[:, i] = hit_tot['scifi_v'][:, 24*i:24*(i+1)].sum(axis=1)
        hit_tot['scifi_v'] = tmp_scifi

        hit_tot['us'] = np.copy(hitmap['us'][i_wall]) # np.sum(hitmap['us'], axis=0, dtype=int)
        hit_tot['ds'] = np.copy(hitmap['ds'][i_wall]) # np.sum(hitmap['ds'], axis=0, dtype=int)
        
        full_min = np.min([np.min(hit) for hit in hit_tot.values()])
        full_max = np.max([np.max(hit) for hit in hit_tot.values()])
        for i, name in enumerate(['scifi_h', 'scifi_v', 'us']):
            im = ax[i].imshow(hit_tot[name][...].T if 'scifi' in name else hit_tot[name][...,0].T, origin='lower', cmap=cmap, vmin=full_min, vmax=full_max, extent=imextent[name+('_h' if name=='us' else '')])
            ax[i].set_title(name)
            cbar = fig.colorbar(im, ax=ax[i])
        plt.suptitle(title+' '+str(i_wall)+' wall')
        #plt.show()
        plt.savefig(PATHF+'f'+str(i_wall)+'.jpg')    
        plt.close()
# In[55]:


plot_grad_walls(hit_wall_grad_zero, title='Ve GRAD ZERO', cmap='RdBu_r')


# In[ ]:




# # SUMMARY AND POTENTIAL NEXT STEPS

# ### Training and testing
# * Model performance seems more or less saturated, yet some small improvements can be achieved with additional training.
# * For further significant changes in the performance modifications in the input, output, training strategy (including loss function) are necessary.
# * Balancing the dataset would also influence the performance, as performance per class now correlates with the class population.
# 
# ### Improving loss function
# * Currently network is trained to equally distinguish everything. However, if we are, for instance, interested in specific class purity (V_e), than the corresponding errors should be penalised more heavily in the loss function.
# 
# ### Event-based analysis
# * This notebook presents comparison of aggregated event visualisations for correct predictions, because there we see a distinctive feature that could be used for separation. This is further confirmed in the event widths analysis.
# * Other type of comparisons can be easily performed. I did not manage to find a clear human-readable feature visible in those distributions (i.e. visualising different types of events misclassified by the algorithm). This does not mean the features are not there, of course. More comparisons can lead to further discoveries.
# 
# ### Broadening the study
# * Additional inputs or outputs can help forcing the network to learn more physically relevant features. Such as:
#     * Adding event energy as the output that network needs to predict.
#     * Adding charge of the hits in the input, so that arrays have not only 0-1 in the pixels.
#     * Trying to predict the axis of the shower (direction).
# 
# ### Relevance maps
# * More work is needed to obtain physical features.
#     * Cross-check the features used to identify wall number (to see that it's intuitively correct).
#     * Average not all events, but events with the same (close) center, so that tails of different showers do not cancel each other.
#     * Compare what is relevant for different classes.
# * Maybe check more methods for the relevance map production, some not gradient-based.
