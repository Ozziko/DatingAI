# -*- coding: utf-8 -*-
"""@author: oz.livneh@gmail.com

* All rights of this project and my code are reserved to me, Oz Livneh.
* Feel free to use - for personal use!
* Use at your own risk ;-)

Image sequence attentive score regression:
    Predicting each profile score based on its entire sequence of images,
    summarizing the images in the (arbitrary size!) sequence by attention.
    
The net input structure is non-trivial in this script:
    Sample=profile. The profiles are batched. However, since each profil has 
    an arbitrary size sequence of images, the PyTorch dataset already returns 
    a batch of images per sample (profile)- so how profiles are batched when 
    each contains an arbitrary size batch of images?
    
    A normal PyTorch dataloader builds each batch by unsqueezing and 
    concatenating equal shape sample items, for example:
        sample['image'].shape=(3,299,299) 
            -> batch['image'].shape=torch.Size([batch_size,3,299,299])
            
    However, here each sample already contains an arbitrary size batch of 
    images. Therefore, I wrote a custom collate function for thePyTorch 
    dataloader that does not unsqueeze and concatenate the dataset image 
    batches, but simply returns a list:
        sample['images batch'].shape=torch.Size(profile_images_number,3,299,299) 
            -> len(batch['images batches list'])=batch_size,
            batch['images batches list'][i].shape=torch.Size(profile_i_images_number,3,299,299) 

<<<This script is for R&D, experiments, debugging>>>
This script can be run entirely, or executed by sections (for example in Spyder).
"""

#%% main parameters
#--------- general -------------
debugging=True # executes the debugging (short) sections, prints results
#debugging=False
torch_manual_seed=0 # integer or None for no seed; for torch reproducibility, as much as possible
#torch_manual_seed=None

#--------- data -------------
data_folder_path=r'D:\My Documents\Dropbox\Python\DatingAI\Data'
images_folder_path=r'D:\My Documents\Dropbox\Python\DatingAI\Data\Images'
df_pickle_file_name='profiles_df.pickle'

#random_transforms=True # random crop, horizontal flip, color jitter etc. - for data augmentation
random_transforms=False
max_dataset_length=100 # if positive: builds a dataset by sampling only max_dataset_length samples from all available data; requires user approval
#max_dataset_length=0 # if non-positive: not restricting dataset length - using all available data
seed_for_dataset_downsampling=0 # integer or None for no seed; for sampling max_dataset_length samples from dataset

validation_ratio=0.3 # validation dataset ratio from total dataset length

#batch_size_int_or_ratio_float=1e-2 # if float: batch_size=round(batch_size_over_dataset_length*len(dataset_to_split))
batch_size_int_or_ratio_float=64 # if int: this is the batch size
data_workers=0 # 0 means no multiprocessing in dataloaders
#data_workers='cpu cores' # sets data_workers=multiprocessing.cpu_count()

shuffle_dataset_indices_for_split=True # dataset indices for dataloaders are shuffled before splitting to train and validation indices
#shuffle_dataset_indices_for_split=False
dataset_shuffle_random_seed=0 # numpy seed for sampling the indices for the dataset, before splitting to train and val dataloaders
#dataset_shuffle_random_seed=None
dataloader_shuffle=True # samples are shuffled inside each dataloader, on each epoch
#dataloader_shuffle=False

#--------- net -------------
architecture_is_a_pretrained_model=False
net_architecture='my simple CNN'

#architecture_is_a_pretrained_model=True
#net_architecture='inception v3'
#net_architecture='resnet18'

#freeze_pretrained_net_weights=True
freeze_pretrained_net_weights=False

loss_name='MSE'

#--------- training -------------
train_model_else_load_weights=True
#train_model_else_load_weights=False # instead of training, loads a pre-trained model and uses it

epochs=3
learning_rate=1e-5
momentum=0.9

lr_scheduler_step_size=1
lr_scheduler_decay_factor=0.9

best_model_criterion='min val epoch MSE' # criterion for choosing best net weights during training as the final weights
return_to_best_weights_in_the_end=True # when training complets, loads weights of the best net, definied by best_model_criterion
#return_to_best_weights_in_the_end=False

training_progress_ratio_to_log_loss=1 # <=1, inter-epoch logging and reporting loss and metrics during training, period_in_batches_to_log_loss=round(training_progress_ratio_to_log_loss*dataset_samples_number['train']/batch_size)
#plot_realtime_stats_on_logging=True # incomplete implementation!
plot_realtime_stats_on_logging=False
#plot_realtime_stats_after_each_epoch=True
plot_realtime_stats_after_each_epoch=False
#plot_loss_in_log_scale=True
plot_loss_in_log_scale=False

#offer_mode_saving=True # offer model weights saving ui after training (only if train_model_else_load_weights=True)
offer_mode_saving=False
models_folder_path='D:\My Documents\Dropbox\Python\DatingAI\Data\Saved Models'

#%% initialization
import logging
logging.basicConfig(format='%(asctime)s %(funcName)s (%(levelname)s): %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger('main logger')
logger.setLevel(logging.INFO)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from time import time
import copy
import PIL
import multiprocessing
if data_workers=='cpu cores':
    data_workers=multiprocessing.cpu_count()

import torch
torch.manual_seed(torch_manual_seed)

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils


class profiles_dataset(torch.utils.data.Dataset):
    def __init__(self,profiles_df,images_folder_path,transform_func=None):
        self.profiles_df=profiles_df
        self.images_folder_path=images_folder_path
        if transform_func!=None:
            self.transform_func=transform_func
        else:
            self.transform_func=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.profiles_df)

    def __getitem__(self,idx):
        profile_series=self.profiles_df.iloc[idx]
        profile_id=profile_series['profile id']
        profile_score=profile_series['score']       
        image_filenames=profile_series['image filenames']
        images_number=len(image_filenames)
        if images_number>0:
            image_tensors_list=[]
            for image_filename in image_filenames:
                image_path=os.path.join(self.images_folder_path,image_filename)
                image_array=PIL.Image.open(image_path)
                image_tensor=self.transform_func(image_array).unsqueeze(0)
                image_tensors_list.append(image_tensor)
            images_batch=torch.cat(image_tensors_list,0)
        else:
            images_batch=torch.tensor(np.nan)
        
        sample={'profile id':profile_id,
                'profile score':profile_score,
                'image filenames':image_filenames,
                'images number':images_number,
                'images batch':images_batch}
        return sample

def training_stats_plot(stats_dict,fig,loss_subplot,MSE_subplot,plot_loss_in_log_scale=False):
    running_stats_df=pd.DataFrame.from_dict(stats_dict['train']['running metrics'],orient='index')
    epoch_train_stats_df=pd.DataFrame.from_dict(stats_dict['train']['epoch metrics'],orient='index')
    epoch_val_stats_df=pd.DataFrame.from_dict(stats_dict['val']['epoch metrics'],orient='index')
    
    loss_subplot.clear() # clearing plot before plotting, to avoid over-plotting
    if len(running_stats_df)>0:
        loss_subplot.plot(running_stats_df['loss per sample'],'-x',label='running train')
    loss_subplot.plot(epoch_train_stats_df['loss per sample'],'k-o',label='epoch train')
    loss_subplot.plot(epoch_val_stats_df['loss per sample'],'r-o',label='epoch val')
    loss_subplot.set_ylabel('loss per sample')
    loss_subplot.set_xlabel('epoch')
    loss_subplot.grid()
    loss_subplot.legend(loc='best')
    if plot_loss_in_log_scale:
        loss_subplot.set_yscale('log')
        
    MSE_subplot.clear() # clearing plot before plotting, to avoid over-plotting
    if len(running_stats_df)>0:
        MSE_subplot.plot(running_stats_df['MSE']**0.5,'-x',label='running train')
    MSE_subplot.plot(epoch_train_stats_df['MSE']**0.5,'k-o',label='epoch train')
    MSE_subplot.plot(epoch_val_stats_df['MSE']**0.5,'r-o',label='epoch val')
    MSE_subplot.set_ylabel('sqrt(MSE)')
    MSE_subplot.set_xlabel('epoch')
    MSE_subplot.grid()
    MSE_subplot.legend(loc='best')
    if plot_loss_in_log_scale:
        MSE_subplot.set_yscale('log')
    fig.canvas.draw()

class remainder_time:
    def __init__(self,time_seconds):
        self.time_seconds=time_seconds
        self.hours=int(time_seconds/3600)
        self.remainder_minutes=int((time_seconds-self.hours*3600)/60)
        self.remainder_seconds=time_seconds-self.hours*3600-self.remainder_minutes*60

logger.info('script initialized')

#%% reading data, unnesting images from profiles_df (creating unnested_images_df)
profiles_df_path=os.path.join(data_folder_path,df_pickle_file_name)
profiles_df=pd.read_pickle(profiles_df_path)

if max_dataset_length>0 and max_dataset_length<len(profiles_df):
    user_data_approval=input('ATTENTION: downsampling is chosen - building a dataset by sampling only max_dataset_length=%d samples from all available data! approve? y/[n] '%(round(max_dataset_length)))
    if user_data_approval!='y':
        raise RuntimeError('user did not approve dataset max_dataset_length sampling!')
    random.seed(seed_for_dataset_downsampling)
    sampled_indices=random.sample(range(len(profiles_df)),max_dataset_length)
    profiles_df=profiles_df.iloc[sampled_indices]

image_num=0
for row_index in range(len(profiles_df)):
    row_series=profiles_df.iloc[row_index,:]
    image_num+=len(row_series['image filenames'])

simplified_columns=[] # original profiles_df names, except for 'score (levels=%d)'%max_level column which is replaced in 'score'
for col in profiles_df.columns:
    if 'score' in col:
        simplified_columns.append('score')
    else:
        simplified_columns.append(col)
profiles_df.columns=simplified_columns

logger.info('comleted reading profiles_df of %.1e profiles, containing %.1e images'%(
        len(profiles_df),image_num))

#%% building a torch dataset
"""torchvision.transforms accept PIL images, and not np images that are 
    created when using skimage as presented in 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial

torchvision transforms: https://pytorch.org/docs/stable/torchvision/transforms.html

ATTENTION: transform_func in this section must include 
    torchvision.transforms.ToTensor()!
"""
if not random_transforms:
    input('random_transforms=False was set, meaning no data augmentation, aknowledge by hitting enter')
if architecture_is_a_pretrained_model:
    if net_architecture=='inception v3':
        input_size_for_pretrained=299
    else:
        input_size_for_pretrained=224
    
    if random_transforms:
        transform_func=torchvision.transforms.Compose([
                torchvision.transforms.Resize(input_size_for_pretrained+10),
                torchvision.transforms.RandomCrop(input_size_for_pretrained),
                torchvision.transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0,hue=0),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # required for pre-trained torchvision models!
                ])
    else:
        transform_func=torchvision.transforms.Compose([
                torchvision.transforms.Resize(input_size_for_pretrained),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # required for pre-trained torchvision models!
                ])
else:
    if random_transforms:
        transform_func=torchvision.transforms.Compose([
    #            torchvision.transforms.Resize(400),
                torchvision.transforms.RandomCrop(390),
                torchvision.transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0,hue=0),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                ])
    else:
        transform_func=torchvision.transforms.Compose([
                torchvision.transforms.Resize(390),
                torchvision.transforms.ToTensor(),
                ])
"""torchvision.transforms.ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the 
    range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the 
    range [0.0, 1.0] if the PIL Image belongs to one of the modes 
    (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has 
    dtype = np.uint8
In the other cases, tensors are returned without scaling
source: https://pytorch.org/docs/stable/torchvision/transforms.html
"""
dataset_w_transform=profiles_dataset(profiles_df=profiles_df,
    images_folder_path=images_folder_path,transform_func=transform_func)

#%% (debugging) verifying dataset by plotting
sample_index_to_verify=0
images_per_row=3
# end of inputs ---------------------------------------------------------------

if debugging:
    sample=dataset_w_transform[sample_index_to_verify]
    images_batch=sample['images batch']
    images_grid=np.transpose(vutils.make_grid(
            images_batch,nrow=images_per_row,padding=5,
            scale_each=True,normalize=True).cpu(),(1,2,0))
    plt.figure()
    plt.subplot(1,1,1)
    plt.axis('off')
    plt.title('profiles_dataset profile %d images'%sample_index_to_verify)
    plt.imshow(images_grid)

#%%

def collate_samples_into_batch_functional(features_to_leave_unstacked):
    def collate_samples_into_batch(samples_list):
        batch={}
        for key in samples_list[0]: # taking the first sample keys since they are the same for all samples
            if key in features_to_leave_unstacked:
                batch.update({key:[sample[key] for sample in samples_list]})
            else:
                batch.update({key:torch.tensor([sample[key] for sample in samples_list])})
        return batch

