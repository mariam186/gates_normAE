import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

#from utils import load_cleaned_dataset
#from utils_model import HRDecoder, HREncoder
#from load_nIDps import create_nIDP_file

#from paper_utils import mask_imgs
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#%% Loading data
main_path="/project_cephfs/3022017.06/projects/marzab/normVAE"
data_path="/project/3022027.01/Gates_old/data/BAMBAM_BMGF/"
out_dir=main_path+ "/results/run1/"
os.makedirs(out_dir,exist_ok=True)
ids_path='/project_cephfs/3022017.06/projects/marzab/normVAE/data/clean_ids.csv'

ids=pd.read_csv(ids_path,index_col=0)
#%% create test and train ids
randomized_ids = ids.sample(frac=1).reset_index(drop=True)


ids_test=randomized_ids[:227]
ids_train=randomized_ids[227:]
ids_test.to_csv(out_dir+'ids_test.csv')#,index=False)
ids_train.to_csv(out_dir+'ids_train.csv')

#%% UTIL1: loading the data

from nilearn import image
data=[]
for idx in ids_train['complete_name']:
    img = nib.load(data_path+idx)
    img = image.resample_img(img, target_affine = np.eye(3)*3)
    data.append(np.expand_dims(np.float32(img.get_fdata()), 0))
data = np.concatenate(data,axis=0)  

x=data[:,2:66,3:75,3:75]
del data


#%% UTIL2: creatimg age_sex_dempgraphic data: run this one time
# nIDP_path=main_path+'/data/clean_nIDPs.csv'
# nIDPs=pd.read_csv(nIDP_path,index_col=0)
# nIDPs=nIDPs.set_index('complete_name')
# nIDPs=nIDPs[['correctedScanAge','BasicFamilyAndChild::childGender','correctedAgeDays']]
# nIDPs=nIDPs.rename(columns={'correctedScanAge':'age','BasicFamilyAndChild::childGender':'sex','correctedAgeDays':'alt_age'})
# nIDPs['age'] = nIDPs['age'].fillna(nIDPs['alt_age'])
# nIDPs=nIDPs.loc[ids['complete_name']]  
# nIDPs.to_csv(main_path+'/data/clean_demo.csv')  
#%% preparing demogoraphic data
nIDPs=pd.read_csv(main_path+'/data/clean_demo.csv')
# Replace 'female' with 0 and 'male' with 1 in the 'sex' column
nIDPs['sex'] = nIDPs['sex'].replace({'Female': 0, 'Male': 1})

# Divide 'age' column by 365 to convert to years
nIDPs['age'] = nIDPs['age'] / 365
nIDPs = nIDPs.drop(columns=['alt_age'])
nIDPs=nIDPs.set_index('complete_name')
y_train=nIDPs.loc[ids_train['complete_name']] 
y_train.to_csv(out_dir+'y_train.csv')
y_test=nIDPs.loc[ids_test['complete_name']] 
y_test.to_csv(out_dir+'y_test.csv')
#%%Data normalization and reshaping
shape=x.shape
x = x.reshape((shape[0], -1))
scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x.T)

scaler_path=out_dir+'scaler.pkl'
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)
    
print(torch.cuda.memory_allocated())
del x
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
x_norm = x_norm.T.reshape(shape) 

#%% data_loader
print(torch.cuda.memory_allocated())
# Convert to tensors
x_norm = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(1)#.to(device)
y = torch.tensor(y_train[['age', 'sex']].values, dtype=torch.float32)#.to(device)
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
# DataLoader setup
batch_size = 32
train_dataset = TensorDataset(x_norm, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
del x_norm
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
#%%
print(torch.cuda.memory_allocated())
#%% models.py: encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, n_features, h_dim, z_dim, stddev=0.05, dropout_level=0.1):
        super(Encoder, self).__init__()
        
        # Gaussian noise layer (as a parameter, but it will behave differently in training)
        self.gaussian_noise = nn.Parameter(torch.randn(n_features) * stddev, requires_grad=False)
        
        # Create a list of convolutional layers with batch normalization and dropout
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        input_channels = n_features[0]  # assuming n_features = (channels, depth, height, width)
        for n_neurons_layer in h_dim:
            self.conv_layers.append(nn.Conv3d(input_channels, n_neurons_layer, kernel_size=3, padding=1))
            self.batch_norms.append(nn.BatchNorm3d(n_neurons_layer))
            self.dropouts.append(nn.Dropout(dropout_level))
            input_channels = n_neurons_layer
        self.bn_latent=nn.BatchNorm1d(z_dim)
        self.flatten = nn.Flatten()
        self.len_flatten=int(np.prod(n_features[1:])/(8**len(h_dim)))
        self.latent_layer = nn.Linear(self.len_flatten*h_dim[-1], z_dim)  # flattening to match dense input size
        self.observed_age = nn.Linear(z_dim, 1)
        self.observed_sex = nn.Linear(z_dim, 1)
        torch.nn.init.xavier_uniform_(self.observed_sex.weight, gain=1.0)  # Small weights
        torch.nn.init.zeros_(self.observed_sex.bias)  # Zero bias

    def forward(self, x):
        # Add Gaussian noise
        x = x + self.gaussian_noise
        
        # Apply Conv3D layers with LeakyReLU, AveragePooling3D, BatchNorm, and Dropout
        for conv, bn, dropout in zip(self.conv_layers, self.batch_norms, self.dropouts):
            x = conv(x)
            x = F.leaky_relu(x)
            x = F.avg_pool3d(x, kernel_size=2, padding=0)  # AveragePooling3D(2, padding='same')
            x = bn(x)
            x = dropout(x)
        
        # Flatten the output and apply the dense latent layer
        x = self.flatten(x)
        latent = self.latent_layer(x)
        latent_sex=self.bn_latent(latent)
        # Compute observed variables: age and sex
        observed_age = F.relu(self.observed_age(latent))
        observed_sex = torch.tanh(self.observed_sex(latent_sex))
        
        observed = torch.cat([observed_age, observed_sex], dim=-1)  # concatenate age and sex
        return latent, observed
#%% models.py:decoder
class Decoder(nn.Module):
    def __init__(self, z_dim, h_dim, latent_dimx=[1, 8, 9, 9], dropout_level=0.1):
        super(Decoder, self).__init__()
        
        # Fully connected layer to reshape the latent vector into a 3D feature map
        self.fc = nn.Linear(z_dim, np.prod(latent_dimx))
        self.latent_dimx=latent_dimx
        # Create a list of deconvolutional layers with batch normalization and dropout
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        input_channels = latent_dimx[0]
        for n_neurons_layer in h_dim:
            self.conv_layers.append(nn.Conv3d(input_channels, n_neurons_layer, kernel_size=3, padding=1))
            self.batch_norms.append(nn.BatchNorm3d(n_neurons_layer))
            self.dropouts.append(nn.Dropout(dropout_level))
            input_channels = n_neurons_layer
        
        # Final Conv3D layer to output the reconstructed 3D image
        self.final_conv = nn.Conv3d(h_dim[-1], 1, kernel_size=3, padding=1)  # Outputting 1 channel
        # Reinitialize observed_sex weights and biases

    def forward(self, x):
        # Fully connected layer to reshape the latent vector into the latent 3D feature map
        x = self.fc(x)
        x = F.leaky_relu(x)
        x = x.view(-1, *self.latent_dimx)  # Reshape into latent_dimx dimensions
        
        # Apply Conv3D layers with LeakyReLU, Upsampling, BatchNorm, and Dropout
        for conv, bn, dropout in zip(self.conv_layers, self.batch_norms, self.dropouts):
            x = conv(x)
            x = F.leaky_relu(x)
            x = F.interpolate(x, scale_factor=2)  # Upsampling3D(2)
            x = bn(x)
            x = dropout(x)
        
        # Final reconstruction layer
        reconstruction = F.relu(self.final_conv(x))
        return reconstruction
#%% make the autencoder networks 
n_features =  (1,)+shape[1:]
# h1_dim = [(1,64),(64,32),(32,16),(16,1)]#encoder number of nodes per layer
# h2_dim=[(16,32),(32,32),(32,32)]#decoder number of nodes per layer

h1_dim=[128,32,16]
h2_dim=[16,32,128]
z_dim = 10# latent nodes
# Define models (need to rewrite hr_encoder_model and hr_decoder_model for PyTorch)
print(torch.cuda.memory_allocated())
encoder = Encoder( n_features,h1_dim, z_dim).to(device)
print(torch.cuda.memory_allocated())
decoder = Decoder(z_dim,h2_dim).to(device)
print(torch.cuda.memory_allocated())

#%% AE parameters
# Optimizer
lambda_ = 1
base_lr = 0.001# learning ratebase_lr = 0.001
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=base_lr)
print(torch.cuda.memory_allocated())

#%%
# Loss functions
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()
bin_loss_fn = nn.BCEWithLogitsLoss()


print(torch.cuda.memory_allocated())
#%%%% train def
l1_lambda=0.01
l2_lambda=0.01

def train_on_batch(batch_x, batch_y):
    # Ensure models are in training mode
    encoder.train()
    decoder.train()

    # Zero out gradients from previous steps
    optimizer.zero_grad()

    # Forward pass
    batch_latent, batch_observed = encoder(batch_x)
    batch_reconstruction = decoder(batch_latent)

    # Calculate reconstruction loss
    recon_loss = mse_loss_fn(batch_x, batch_reconstruction)

    # Calculate supervised losses
    mae_loss = mae_loss_fn(batch_y[:, :1], batch_observed[:, :1])  # For age
    bin_loss = bin_loss_fn(batch_y[:, 1:], batch_observed[:, 1:])  # For sex (binary classification)
    l1_reg = sum(l1_lambda * torch.sum(torch.abs(param))
    for name, param in encoder.named_parameters()
    if param.requires_grad and "bias" not in name)
    l2_reg = sum(l2_lambda * torch.sum(param ** 2)
    for name, param in encoder.named_parameters()
    if param.requires_grad and "bias" not in name)
    # Weighted combined loss
    #supervised_loss = mae_loss + bin_loss
    ae_loss = lambda_ * 100 * recon_loss +  mae_loss + lambda_ * 100 *bin_loss+ l1_reg + l2_reg


    # Backward pass and optimization step
    ae_loss.backward()
    optimizer.step()

    # Clean up unnecessary tensors to save memory
    del batch_reconstruction, batch_x

    return ae_loss.item(), recon_loss.item(), mae_loss.item(), bin_loss.item()

#%%

def validate_on_batch(batch_x, batch_y):
    # Ensure models are in evaluation mode
    encoder.eval()
    decoder.eval()

    # Forward pass without gradients
    with torch.no_grad():
        batch_latent, batch_observed = encoder(batch_x)
        batch_reconstruction = decoder(batch_latent)

        # Calculate reconstruction loss
        recon_loss = mse_loss_fn(batch_x, batch_reconstruction)

        # Calculate supervised losses
        mae_loss = mae_loss_fn(batch_y[:, :1], batch_observed[:, :1])  # For age
        bin_loss = bin_loss_fn(batch_y[:, 1:], batch_observed[:, 1:])  # For sex (binary classification)

        # Weighted combined loss
        #supervised_loss = mae_loss + bin_loss
        ae_loss = lambda_ * 100 * recon_loss + mae_loss +lambda_ * 100 *  bin_loss

    return ae_loss.item(), recon_loss.item(), mae_loss.item(), bin_loss.item()


#%% Training and val
import pandas as pd

# Create a DataFrame to log losses
loss_df = pd.DataFrame(columns=["Epoch", 
                                "Train_Recon_Loss", "Train_MAE_Loss", "Train_Bin_Loss", "Train_Total_Loss",
                                "Val_Recon_Loss", "Val_MAE_Loss", "Val_Bin_Loss", "Val_Total_Loss"])

n_epochs =20
validation_ratio = 0.05  # Percentage of training batches used for validation

for epoch in range(n_epochs):
    total_train_recon_loss = 0
    total_train_mae_loss = 0
    total_train_bin_loss = 0
    total_val_recon_loss = 0
    total_val_mae_loss = 0
    total_val_bin_loss = 0

    # Training loop with validation
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if i < len(train_loader) * (1 - validation_ratio):  # Training portion
            encoder.train()
            decoder.train()
            
            # Perform training
            ae_loss, recon_loss, mae_loss, bin_loss = train_on_batch(batch_x, batch_y)
            total_train_recon_loss += recon_loss
            total_train_mae_loss += mae_loss
            total_train_bin_loss += bin_loss
        else:  # Validation portion
            val_loss,val_recon_loss,val_mae_loss,val_bin_loss=validate_on_batch(batch_x, batch_y)
            # encoder.eval()
            # decoder.eval()

            # with torch.no_grad():
            #     # Perform validation
            #     batch_latent, batch_observed = encoder(batch_x)
            #     batch_reconstruction = decoder(batch_latent)

            #     val_recon_loss = mse_loss_fn(batch_x, batch_reconstruction).item()
            #     val_mae_loss = mae_loss_fn(batch_y[:, :1], batch_observed[:, :1]).item()
            #     val_bin_loss = bin_loss_fn(batch_y[:, 1:], batch_observed[:, 1:]).item()

            total_val_recon_loss += val_recon_loss
            total_val_mae_loss += val_mae_loss
            total_val_bin_loss += val_bin_loss

    # Calculate averages
    num_train_batches = len(train_loader) * (1 - validation_ratio)
    num_val_batches = len(train_loader) * validation_ratio

    avg_train_recon_loss = total_train_recon_loss / num_train_batches
    avg_train_mae_loss = total_train_mae_loss / num_train_batches
    avg_train_bin_loss = total_train_bin_loss / num_train_batches
    avg_train_total_loss = avg_train_recon_loss + avg_train_mae_loss + avg_train_bin_loss

    avg_val_recon_loss = total_val_recon_loss / num_val_batches
    avg_val_mae_loss = total_val_mae_loss / num_val_batches
    avg_val_bin_loss = total_val_bin_loss / num_val_batches
    avg_val_total_loss = avg_val_recon_loss + avg_val_mae_loss + avg_val_bin_loss

    # Log epoch losses to the DataFrame
    loss_df = pd.concat([loss_df, pd.DataFrame([{
        "Epoch": epoch + 1,
        "Train_Recon_Loss": avg_train_recon_loss,
        "Train_MAE_Loss": avg_train_mae_loss,
        "Train_Bin_Loss": avg_train_bin_loss,
        "Train_Total_Loss": avg_train_total_loss,
        "Val_Recon_Loss": avg_val_recon_loss,
        "Val_MAE_Loss": avg_val_mae_loss,
        "Val_Bin_Loss": avg_val_bin_loss,
        "Val_Total_Loss": avg_val_total_loss
    }])], ignore_index=True)

    # Print epoch summary
    print(f"Epoch [{epoch + 1}/{n_epochs}]")
    print(f"  Train - Recon: {avg_train_recon_loss:.4f}, MAE: {avg_train_mae_loss:.4f}, Bin: {avg_train_bin_loss:.4f}, Total: {avg_train_total_loss:.4f}")
    print(f"  Val   - Recon: {avg_val_recon_loss:.4f}, MAE: {avg_val_mae_loss:.4f}, Bin: {avg_val_bin_loss:.4f}, Total: {avg_val_total_loss:.4f}")

# Save the DataFrame to a CSV file
loss_df.to_csv(out_dir+"training_validation_losses.csv", index=False)

#%%
# Save models and other results
torch.save(encoder.state_dict(), out_dir+'model_encoder.pth')
torch.save(decoder.state_dict(), out_dir+'model_decoder.pth')
#%%
from nilearn import plotting
import numpy as np
from matplotlib import pyplot as plt
def display_nifti(img,affine=np.eye(4),cut_coords=None):
    img=np.squeeze(img.detach().cpu().numpy())
    nifti_image = nib.Nifti1Image(img, affine=affine)
    plotting.plot_img(nifti_image,cut_coords=cut_coords)
    plt.show()    

#the distribution of x and xhat loss     


#%% TEST
def save_imgs(x,file_name='outfile.nii'):
    x=x.squeeze().cpu().detach().numpy()
    x = np.transpose(x, (1, 2, 3, 0))
    nifti_img = nib.Nifti1Image(x, affine=np.eye(4))
    nib.save(nifti_img, file_name)
    return
#%%
save_imgs(batch_x[0:5],out_dir+'orig.nii')
x1=decoder(encoder(batch_x[0:5])[0])
save_imgs(x1,out_dir+'rec.nii')
    
    
    
#x1=decoder(encoder(batch_x[0:3])[0])


