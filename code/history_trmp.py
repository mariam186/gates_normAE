#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:50:54 2024

@author: marzab
"""
nIDPs=nIDPs_def(ids_table)
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
runcell(9, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Define optimizers', '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Loss functions', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(4, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('cross_validation whitin the batch', '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Define optimizers', '/project/3022027.01/Gates/codes/ae_2.py')
recon_loss

## ---(Wed Nov 27 19:21:38 2024)---
runcell(0, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(2, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Loss functions', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(4, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('cross_validation whitin the batch', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(6, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(8, '/project/3022027.01/Gates/codes/ae_2.py')

## ---(Fri Nov 29 16:37:05 2024)---
runcell(0, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(2, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Loss functions', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(4, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('cross_validation whitin the batch', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(6, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(7, '/project/3022027.01/Gates/codes/ae_2.py')
dfx=pd.read_csv('/project/3022027.01/Gates/data/clean_nIDPs.csv',index_col=0)
dfx
ids=pd.read_csv('/project/3022027.01/Gates/data/ids/ids_00.csv')
ids
from utils import jonny_ids
idsx=jonny_ids()
paper_dir='/project/3022027.01/Gates/results/paper/'
umap_dir='/project/3022027.01/Gates/results/run_1/kf_01/umap.csv'
latent_umap_df=pd.read_csv(umap_dir)
runcell(0, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Loss functions', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(4, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('cross_validation whitin the batch', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(6, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(7, '/project/3022027.01/Gates/codes/ae_2.py')
df=pd.read_csv('/project/3022027.01/Gates/data/nIDP/clean_nIDPs.csv')
df.columns
df1=df[['name',
       'BasicFamilyAndChild::childGender']]
df=pd.read_csv('/project/3022027.01/Gates/data/nIDP/clean_nIDPs.csv')
df1=df[['name',
       'BasicFamilyAndChild::childGender']]
df=pd.read_csv('/project/3022027.01/Gates/data/clean_nIDPs.csv')
df1=df[['name',
       'BasicFamilyAndChild::childGender']]
unique_counts = df1.drop_duplicates(subset='name').groupby('BasicFamilyAndChild::childGender')['name'].count()
df1['names'] = df1['name'].str.extract(r'(sub-\w+)')
df1['names'] = df1['name'].str.extract(r'(sub_\w+)')
df1['names'] = df1['name'].str.extract(r'(sub-\w+)')
unique_counts = df1.drop_duplicates(subset='names').groupby('BasicFamilyAndChild::childGender')['names'].count()
251/(251/313)
251/(251+313)
runfile('/project/3022027.01/Gates/codes/ae_2.py', wdir='/project/3022027.01/Gates/codes')
runcell(8, '/project/3022027.01/Gates/codes/ae_2.py')

## ---(Sun Nov 24 18:01:54 2024)---
runcell(0, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(2, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(4, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('cross_validation whitin the batch', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(6, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(7, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(0, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(8, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(6, '/project/3022027.01/Gates/codes/ae_2.py')
nIDPs=nIDPs_def(ids_table)
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
runcell(9, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Define optimizers', '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Loss functions', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(4, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('cross_validation whitin the batch', '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Define optimizers', '/project/3022027.01/Gates/codes/ae_2.py')
recon_loss
runcell(8, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(9, '/project/3022027.01/Gates/codes/ae_2.py')
runcell(10, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Define optimizers', '/project/3022027.01/Gates/codes/ae_2.py')
batch_x.shape
batch_y.shape
batch_y
a1,b1=encoder(batch_x)
b1
a1
a1.shape
c=decoder(b1)
c=decoder(a1)
decoder
decoder.summary()
decoder = hr_decoder_model(z_dim , h2_dim,latent_dimx=[8,9,9,16])
decoder.summary()
decoder = hr_decoder_model(z_dim , h2_dim,latent_dimx=[7,8,8,16])
decoder.summary()
decoder = hr_decoder_model(z_dim , h2_dim,latent_dimx=[8,9,9,16])
decoder.summary()
from utils_modelss import make_decoder_model_joint,make_encoder_model_joint,hr_decoder_model,hr_encoder_model
from load_nIDps import create_nIDP_file
runcell(4, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('cross_validation whitin the batch', '/project/3022027.01/Gates/codes/ae_2.py')
runcell(10, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Define optimizers', '/project/3022027.01/Gates/codes/ae_2.py')
batch_x.shape
runcell(2, '/project/3022027.01/Gates/codes/ae_2.py')
runcell('Define optimizers', '/project/3022027.01/Gates/codes/ae_2.py')
mask_path='/project/3022027.01/Gates/data/Jonny/mask.nii'
mask=nib.load(mask_path)
mask=np.float32(mask.get_fdata())
mask = mask.resample_img(mask, target_affine = np.eye(3)*3)
mask=nib.load(mask_path)
from paper_utils import mask_imgs
batch_latent, batch_observed = encoder(batch_x)
batch_reconstruction = decoder(batch_latent)
runcell(0, '/project/3022027.01/Gates/codes/ae_2.py')