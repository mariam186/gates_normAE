#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 19:10:36 2022

@author: marzab
"""
import os
os.chdir('/project_cephfs/3022017.06/projects/marzab')
#%%
import time
from pathlib import Path
import numpy as np
import pickle

from sklearn.preprocessing import  MinMaxScaler

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import optimizers as Optimizers
import nibabel as nib

from utils import load_cleaned_dataset
from utils_models import make_decoder_model_joint,make_encoder_model_joint,hr_decoder_model,hr_encoder_model
from load_nIDps import create_nIDP_file
print(tf.config.list_physical_devices('GPU'))
#tf.config.run_functions_eagerly(True)
import random
#%%
#%%
def mask_imgs(imgx,mask_path='/project/3022027.01/Gates/data/Jonny/mask.nii',resamp=False,crop=True):
    mask=nib.load(mask_path)
    mask=np.float32(mask.get_fdata())
    if resamp:
        mask = mask.resample_img(mask, target_affine = np.eye(3)*3)
    if crop:
        if resamp:
            mask=mask[2:66,3:75,3:75]
        else:
            mask=mask[5:149,2:178,2:178]    
    
    mask=tf.convert_to_tensor(mask)
    batch_size = imgx.get_shape().as_list()[0]
    
    #print("================")
    
    #print(imgx.shape[0])
    #print(batch_size)
    mask=tf.repeat(mask[np.newaxis,...],batch_size,axis=0)
    x_masked=tf.boolean_mask(imgx,(mask!=0))
    x_masked=tf.reshape(x_masked,[batch_size,-1])
    
    #mask=tf.convert_to_tensor(mask)
    
    #mask=tf.repeat(mask[np.newaxis,...],imgx.shape[0],axis=0)
    #print("================")
    

    
    #x_masked=tf.boolean_mask(imgx,(mask!=0))
    #print(x_masked.shape)
    
    #print("================")
    #x_masked=tf.reshape(x_masked,[imgx.shape[0],-1])
    
    return x_masked
#%%# Loss functions
# Reconstruction cost
mse_loss_fn = tf.keras.losses.MeanSquaredError()
# Supervised cost
mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
bin_loss_fn= tf.keras.losses.BinaryCrossentropy(from_logits=False)
#evalution_metrics - not minimizing during the training 
mae_age_fn=tf.keras.metrics.MeanAbsoluteError()
acc_sex_fn = tf.keras.metrics.BinaryAccuracy()
mse_rec_fn = tf.keras.metrics.MeanSquaredError()
#%%
# Training step
#@tf.function
def train_on_batch(batch_x, batch_y):
    with tf.GradientTape() as tape:
        
        acc_sex_fn.reset_states()
        mae_age_fn.reset_states()
        # Inference
        batch_latent, batch_observed = encoder(batch_x)
        batch_reconstruction = decoder(batch_latent)
        
        masked_recon=mask_imgs(batch_reconstruction)
        #print("mask is done")
        masked_x=mask_imgs(batch_x)
        #print("mask2 is done")
        # Loss functions
        #recon_loss = mse_loss_fn(batch_x, batch_reconstruction)
        recon_loss = mse_loss_fn(masked_x, masked_recon)
        mae_loss = mae_loss_fn(batch_y[:,:1], batch_observed[:,:1])
        bin_loss=bin_loss_fn(batch_y[:,1:], batch_observed[:,1:])
        
        supervised_loss=mae_loss +bin_loss
        ae_loss = lambda_*10*recon_loss + (1-lambda_)*supervised_loss
        #print("this is important")
        
        acc_sex=acc_sex_fn(batch_y[:,1:], batch_observed[:,1:])
        mae_age=mae_age_fn(batch_y[:,:1], batch_observed[:,:1])
    gradients = tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return recon_loss, supervised_loss, acc_sex,mae_age

#%% cross_validation whitin the batch
def test_on_batch(batch_tx, batch_ty):
    acc_sex_fn.reset_states()
    mae_age_fn.reset_states()
    mse_rec_fn.reset_state()
    # Inference
    batch_latent, batch_observed = encoder(batch_tx)
    batch_reconstruction = decoder(batch_latent)
    masked_recon=mask_imgs(batch_reconstruction)
    masked_tx=mask_imgs(batch_tx)
    # Loss functions
    #recon_loss = mse_loss_fn(batch_x, batch_reconstruction)
    recon_loss = mse_rec_fn(masked_tx, masked_recon)
    #recon_loss = mse_rec_fn(batch_tx, batch_reconstruction)


    acc_sex=acc_sex_fn(batch_ty[:,1:], batch_observed[:,1:])
    mae_age=mae_age_fn(batch_ty[:,:1], batch_observed[:,:1])
    supervised_loss=mae_age +1-acc_sex
    return recon_loss, supervised_loss, acc_sex,mae_age


#%%

def nIDPs_def(ids_table,nIDPs_path='/home/mrstats/marzab/project/Gates/data/nIDP/clean_nIDPs.csv'):
    #% Loading age and sex 
    
    nIDPs=pd.read_csv(nIDPs_path,index_col=0)
    
    nIDPs=nIDPs.set_index('complete_name')
    nIDPs=nIDPs.loc[ids_table['complete_name']]
    
    nIDPs=nIDPs[['correctedScanAge','BasicFamilyAndChild::childGender','correctedAgeDays']]
    nIDPs=nIDPs.rename(columns={'correctedScanAge':'age','BasicFamilyAndChild::childGender':'sex','correctedAgeDays':'alt_age'})
    
    #check for nan in age 
    ag1=nIDPs['age'].values
    ag2=nIDPs['alt_age'].values
    ag1[np.isnan(ag1)]=ag2[np.isnan(ag1)]
    ## somehow it does fix nIDPs[age] by assiging directly!! 
    age=nIDPs['age'].values[:, np.newaxis].astype('float32')/356
    sex_enc=pd.get_dummies(nIDPs['sex'])
    sex=sex_enc['Male'].values[:, np.newaxis].astype('float32')
    
    #y_data= np.concatenate((age, sex), axis=1).astype('float32')
    nIDPs['sex']=sex
    nIDPs['age']=age
    return nIDPs
#%%
# kf=0

# kf_name='kf_{:02d}'.format(kf)
# print(kf_name)
# out_dir=os.path.join("/home/mrstats/marzab/project/Gates/results/run_4/",kf_name)


out_dir="/home/mrstats/marzab/project/Gates/results/run_10/"
os.makedirs(out_dir,exist_ok=True)
#ids_name='ids_{:02d}'.format(kf)+'.csv'
#ids=pd.read_csv('/home/mrstats/marzab/project/Gates/data/ids/'+ids_name)
#load the data
from utils import jonny_ids
ids=jonny_ids()
random.shuffle(ids)
ids_test=ids[:200]
id_test=pd.DataFrame(data=ids_test,columns=["complete_name"])
id_test.to_csv(out_dir+'ids_test.csv')#,index=False)
ids=ids[200:900]
#%%
data_path='/project/3022027.01/Gates/data/Jonny/BAMBAM_BMGF/'
create_nIDP_file(data_path=data_path)
x,ids_table=load_cleaned_dataset(ids=ids,data_path=data_path,crop=True,resamp=False)#ids=ids['complete_name']
nIDPs=nIDPs_def(ids_table)
#% creating output repositories    
model_dir=os.path.join(out_dir,'models')

os.makedirs(model_dir,exist_ok=True)

shape=x.shape
x=x.reshape((shape[0],-1))
scaler = MinMaxScaler()
x_T = scaler.fit_transform(x.T) 

x_norm=np.reshape(x_T.T,shape) 
#%preparing dataset
batch_size = 10
n_samples = x_norm.shape[0]
x_train=x_norm.reshape(x_norm.shape+(1,))
del x,x_norm
#%%
input_shape=x_train.shape
#%Create the dataset iterator
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,nIDPs[['age','sex']].values))
train_dataset = train_dataset.shuffle(buffer_size=n_samples)
train_dataset = train_dataset.batch(batch_size)
print(x_train.shape)

n_features = input_shape[1:]
h1_dim = [64,32,16]#encoder number of nodes per layer
h2_dim=[16,32,32]#decoder number of nodes per layer
z_dim = 50 # latent nodes
base_lr = 0.001# learning rate
#% make the networks
encoder = hr_encoder_model(n_features, h1_dim, z_dim)
decoder = hr_decoder_model(z_dim , h2_dim,latent_dimx=[9,11,11,16])
# encoder = make_encoder_model_joint(n_features, h1_dim, z_dim)
# decoder = make_decoder_model_joint(z_dim , h2_dim)
lambda_ = .5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(base_lr,decay_steps=10000,decay_rate=0.96,staircase=True)

df = pd.DataFrame(columns=['epoch','ETA','Reconstruction cost','Supervised cost','sex_acc','age_mae'])
df_cross_validation=pd.DataFrame(columns=['epoch','ETA','Reconstruction cost','Supervised cost','sex_acc','age_mae'])
#%%# Define optimizers
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1.)
number_batch_cv=n_samples/batch_size-2 # the last 2 batchs will be used as CV (it's fixed batches through iterations)
# Training loop
n_epochs=5
start = time.time()
for epoch in range(n_epochs):
    

    # Functions to calculate epoch's mean performance
    epoch_recon_loss_avg = tf.metrics.Mean()
    epoch_supervised_loss_avg = tf.metrics.Mean()    

    epoch_acc_sex_avg=tf.metrics.Mean()
    epoch_mae_loss_avg=tf.metrics.Mean()
    # Functions to calculate epoch's mean performance in TEST
    epoch_recon_loss_test_avg = tf.metrics.Mean()
    epoch_supervised_loss_test_avg = tf.metrics.Mean()

    epoch_acc_sex_test_avg=tf.metrics.Mean()
    epoch_mae_loss_test_avg=tf.metrics.Mean()
    
    #counter=0
    for batch, (batch_x, batch_y) in enumerate(train_dataset):

        if batch < number_batch_cv:
            #print("this train")
            recon_loss, supervised_loss, acc_sex,mae_age = train_on_batch(batch_x, batch_y)
            #print(recon_loss)
            epoch_recon_loss_avg(recon_loss)
            epoch_supervised_loss_avg(supervised_loss)
            epoch_acc_sex_avg(acc_sex)
            epoch_mae_loss_avg(mae_age)       
            epoch_time = time.time() - start

        else:
            #print("this is test")
            recon_loss_test, supervised_loss_test,acc_sex_test,mae_age_test = test_on_batch(batch_x, batch_y)
            epoch_recon_loss_test_avg(recon_loss_test)
            epoch_supervised_loss_test_avg(supervised_loss_test)
            epoch_acc_sex_test_avg(acc_sex_test)
            epoch_mae_loss_test_avg(mae_age_test)

        #counter=counter+1
    print('{:3d}: {:.2f}s ETA: {:.2f}s  Reconstruction cost: {:.4f}  Supervised cost: {:.4f}   sex_acc:{:.2f} age_mae: {:.4f}'
          .format(epoch + 1, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_recon_loss_avg.result(),
                  epoch_supervised_loss_avg.result(),
                  epoch_acc_sex_avg.result(),
                  epoch_mae_loss_avg.result()))
    df.loc[epoch,['epoch']]=epoch + 1
    df.loc[epoch,['ETA']]=epoch_time * (n_epochs - epoch)
    df.loc[epoch,['Reconstruction cost']]=epoch_recon_loss_avg.result()
    df.loc[epoch,['Supervised cost']]=epoch_supervised_loss_avg.result()
    df.loc[epoch,['sex_acc']]=epoch_acc_sex_avg.result()
    df.loc[epoch,['age_mae']]=epoch_mae_loss_avg.result()

    
    print('{:3d}: {:.2f}s ETA: {:.2f}s  Reconstruction CV: {:.4f}  Supervised cost CV: {:.4f}  sex_acc:{:.2f} age_mae: {:.4f}'
          .format(epoch + 1, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_recon_loss_test_avg.result(),
                  epoch_supervised_loss_test_avg.result(),
                  epoch_acc_sex_test_avg.result(),
                  epoch_mae_loss_test_avg.result()))
    df_cross_validation.loc[epoch,['epoch']]=epoch + 1
    df_cross_validation.loc[epoch,['ETA']]=epoch_time * (n_epochs - epoch)
    df_cross_validation.loc[epoch,['Reconstruction cost']]=epoch_recon_loss_test_avg.result()
    df_cross_validation.loc[epoch,['Supervised cost']]=epoch_supervised_loss_test_avg.result()
    df_cross_validation.loc[epoch,['sex_acc']]=epoch_acc_sex_test_avg.result()
    df_cross_validation.loc[epoch,['age_mae']]=epoch_mae_loss_test_avg.result()
    
    print('================================================================================================')       
 #%
df_cross_validation.to_csv(model_dir+'/loss_cv.csv')
df.to_csv(model_dir+'/loss_train.csv')
encoder.save(model_dir +'/model_encoder.h5')
decoder.save(model_dir+'/model_decoder.h5')
nIDPs.to_csv(out_dir+'/train_table.csv')

time3=time.time()
print('overall proccesing time :',time3-start)
#%%
# latent_dim=[]
# for d in range(50):
#     latent_dim.append('d_'+str(d))
# for batch, (batch_x, batch_y) in enumerate(train_dataset):    
#     latent_nodes,observed_train =encoder.predict(batch_x) 
        
#     data=pd.DataFrame(columns=latent_dim,data=latent_nodes)
#     nIDPs_batch=nIDPs.reset_index()
#     latent_df=pd.concat([nIDPs,data],axis=1)
#%%
nIDPs=nIDPs.reset_index()
latent_dim=[]
for d in range(50):
    latent_dim.append('d_'+str(d))
for i in range(0,n_samples,batch_size):   
    len_i=np.min([n_samples,i+batch_size])
    latent_nodes,observed_train =encoder.predict(x_train[i:len_i]) 
    
    data=pd.DataFrame(columns=latent_dim,data=latent_nodes)
    nIDPs_temp=nIDPs[i:len_i]
    nIDPs_temp=nIDPs_temp.reset_index(drop=True)
    latent_df_temp=pd.concat([nIDPs_temp,data],axis=1)
    if i==0:
        latent_df=latent_df_temp
    else:
        latent_df=pd.concat([latent_df,latent_df_temp],axis=0)
latent_df=latent_df.reset_index(drop=True)    
latent_df.to_csv(out_dir+'/latent_train.csv')

print('*****************************************************************************************')
#-           
#%%
# latent_dim=[]
# for d in range(50):
#     latent_dim.append('d_'+str(d))
    
# latent_nodes,observed_train =encoder.predict(x_train) 
    
# data=pd.DataFrame(columns=latent_dim,data=latent_nodes)
# nIDPs=nIDPs.reset_index()
# latent_df=pd.concat([nIDPs,data],axis=1)
# latent_df.to_csv(out_dir+'/latent_train.csv')

# print('*****************************************************************************************')
# #-       




























