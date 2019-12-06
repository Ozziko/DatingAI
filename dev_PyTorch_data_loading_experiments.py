# -*- coding: utf-8 -*-
"""@author: oz.livneh@gmail.com

* All rights of this project and my code are reserved to me, Oz Livneh.
* Feel free to use - for personal use!
* Use at your own risk ;-)

<<<This script is for R&D, experiments, debugging!>>>
It is meant to be executed by sections (for example in Spyder).
"""

#%% initialization
import logging
logging.basicConfig(format='%(asctime)s %(funcName)s (%(levelname)s): %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger('data processing logger')
logger.setLevel(logging.INFO)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

import PIL
from skimage import io as skimage_io
from skimage import transform as skimage_transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def plot_df_images_w_scores(index_iterator_to_plot,df,images_folder_path,
                            images_per_row=4):
    """created on 2019/02/22 to plot images with their profile index and score
        
    df is the dataframe for profiles, assumed to contain columns:
        'profile index','score (levels=%d)'%score_levels,'image filename'
    """
    assert images_per_row>0, 'images_per_row must be strictly positive'
    # finding the score column
    score_column_idx=None
    for i,column in enumerate(df.columns):
        if 'score' in column:
            score_column_idx=i
            break
    if score_column_idx==None:
        raise RuntimeError("no existing column in df contains 'score'!")
    
    plt.figure()
    columns_num=math.ceil(len(index_iterator_to_plot)/images_per_row)
    for i in index_iterator_to_plot:
        df_row=df.iloc[i,:]
        profile_index=df_row['profile index']
        profile_score=df_row.iloc[score_column_idx]
        image_filename=df_row['image filename']
        
        plt.subplot(columns_num,images_per_row,i+1)
        image_array=plt.imread(os.path.join(images_folder_path,image_filename))
        plt.imshow(image_array)
        plt.title('profile index: %s, score: %d'%(profile_index,profile_score))
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    plt.show()

class unnested_np_images_dataset(Dataset):
    def __init__(self,images_df,images_folder_path,transform_func=None):
        self.images_df=images_df
        self.images_folder_path=images_folder_path
        self.transform_func=transform_func

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,idx):
        image_path=os.path.join(self.images_folder_path,
                                self.images_df['image filename'].iloc[idx])
        profile_score=self.images_df['score'].iloc[idx]
        profile_score_code=self.images_df['score code'].iloc[idx]
        profile_index=self.images_df['profile index'].iloc[idx]
        image_array=skimage_io.imread(image_path)
        
        if self.transform_func!=None:
            image_array=self.transform_func(image_array)
        
        sample={'image array':image_array,'profile score':profile_score,
                'profile score code':profile_score_code,'profile index':profile_index}
        return sample

class unnested_PIL_images_dataset(Dataset):
    def __init__(self,images_df,images_folder_path,transform_func=None):
        self.images_df=images_df
        self.images_folder_path=images_folder_path
        self.transform_func=transform_func

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,idx):
        image_path=os.path.join(self.images_folder_path,
                                self.images_df['image filename'].iloc[idx])
        profile_score=self.images_df['score'].iloc[idx]
        profile_score_code=self.images_df['score code'].iloc[idx]
        profile_index=self.images_df['profile index'].iloc[idx]
        image_array=PIL.Image.open(image_path)
        
        if self.transform_func!=None:
            image_array=self.transform_func(image_array)
        
        sample={'image array':image_array,'profile score':profile_score,
                'profile score code':profile_score_code,'profile index':profile_index}
        return sample

def plot_dataset_images_w_scores(index_iterator_to_plot,dataset,images_per_row,
                                 image_format='np'):
    plt.figure()
    columns_num=math.ceil(len(index_iterator_to_plot)/images_per_row)
    for i in index_iterator_to_plot:
        sampe=dataset[i]
        image_array=sampe['image array']
        image_score=sampe['profile score']
        profile_index=sampe['profile index']
        
        plt.subplot(columns_num,images_per_row,i+1)
        if image_format=='np->torch': # to return from a torch format that reached from a np format, to a np for plotting. see # Helper function to show a batch from https://pytorch.org/tutorials/beginner/data_loading_tutorial
            image_array=image_array.transpose((1,2,0))
        elif image_format=='PIL->torch': # to return from a torch format that reached from a PIL format, to a np for plotting. see # Helper function to show a batch from https://pytorch.org/tutorials/beginner/data_loading_tutorial
            image_array=image_array.numpy().transpose((1,2,0))
        plt.imshow(image_array)
        plt.title('profile index: %s, score: %d'%(profile_index,image_score))
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])

class Rescale():
    """adjusted from https://pytorch.org/tutorials/beginner/data_loading_tutorial
    Rescale an image sample to a output_size if necessary 
        (if output_size is different than its original size)

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size=output_size

    def __call__(self,image_array):
        h,w=image_array.shape[:2]
        if isinstance(self.output_size,int):
            if h>w:
                new_h,new_w=self.output_size*h/w,self.output_size
            else:
                new_h,new_w=self.output_size,self.output_size* w/h
        else:
            new_h,new_w=self.output_size
        new_h,new_w=int(new_h),int(new_w)
        
        if new_h==h and new_w==w:
            return image_array
        else:
            image_array=skimage_transform.resize(image_array,(new_h,new_w))
            return image_array

class RandomCrop():
    """adjusted from https://pytorch.org/tutorials/beginner/data_loading_tutorial
    Crop randomly the image in a sample.

    Args:
        sample: a dictionary containint 'image array' key, storing the image 
            array to transform
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size=(output_size,output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size

    def __call__(self,image_array):
        h, w = image_array.shape[:2]
        new_h,new_w = self.output_size

        top=np.random.randint(0, h - new_h)
        left=np.random.randint(0, w - new_w)

        image_array=image_array[top:top+new_h, left:left +new_w]
        
        return image_array

class ToTensor():
    """adjusted from https://pytorch.org/tutorials/beginner/data_loading_tutorial
    Convert image_array from a numpy array to a torch Tensor -
        swaps color axis because numpy image is [H x W x C] and 
        a torch image is [C X H X W]
    """
    def __call__(self,image_array):
        return image_array.transpose((2,0,1))

logger.info('initialized')
#%% reading data, unnesting images from raw_df (creating unnested_images_df)
""" 'profile index' in unnested_images_df is the row index in raw_df
"""
data_folder_path='D:\My Documents\Dropbox\Python\DatingAI\Data'
images_folder_path='D:\My Documents\Dropbox\Python\DatingAI\Data\Images'
# end of inputs ---------------------------------------------------------------
raw_df_path=os.path.join(data_folder_path,'raw_df.pickle')
raw_df=pd.read_pickle(raw_df_path)

# finding score column (column name in the format of 'score (levels=%d)'%score_levels)
score_column_name=None
for column in raw_df.columns:
    if 'score' in column:
        score_column_name=column
        break
if score_column_name==None:
    raise RuntimeError("no existing column name in raw_df contains 'score'!")

# translating scores to score codes
scores_set=set(raw_df[score_column_name])
score_to_code_dict={}
code_to_score_dict={}
for code,score in enumerate(sorted(scores_set)):
    score_to_code_dict.update({score:code})
    code_to_score_dict.update({code:score})

# unnesting images
#unnested_images_dict={} # = {filename:{'score':...,'score code':...,'profile id':...}}
unnested_images_df_list=[] # = ['profile index','profile id','score','score code','image filename']
for row_index in range(len(raw_df)):
    row_series=raw_df.iloc[row_index,:]
    profile_id=row_series['profile id']
    profile_score=row_series[score_column_name]
    profile_score_code=score_to_code_dict[profile_score]
    for filename in row_series['image filenames']:
#        if filename in unnested_images_dict:
#            if filename=='pq_400.pn':
#                logger.info("default missing image '%s' appears in profile id %s and %s, over-writing by the last appearance"%(
#                        filename,unnested_images_dict[filename]['profile id'],profile_id))
#            else:
#                logger.warning("image '%s' appears in profile id %s and %s, over-writing by the last appearance"%(
#                        filename,unnested_images_dict[filename]['profile id'],profile_id))
#        unnested_images_dict.update({filename:{'score':profile_score,
#                        'score code':profile_score_code,'profile id':profile_id}})
        unnested_images_df_list.append([row_index,profile_id,profile_score,
                                     profile_score_code,filename])
unnested_images_df=pd.DataFrame(unnested_images_df_list,
                columns=['profile index','profile id','score','score code',
                         'image filename'])
#%% building a torch dataset of unnested np images and testing it
index_iterator_to_plot=range(20)
images_per_row=4
# end of inputs ---------------------------------------------------------------

dataset=unnested_np_images_dataset(images_df=unnested_images_df,
    images_folder_path=images_folder_path,transform_func=None)

# plotting from dataset
plot_dataset_images_w_scores(index_iterator_to_plot,dataset,images_per_row,
                             image_format='np')
plt.suptitle('plotting data from the dataset')

# verifying against directly plotting the images by plot_df_images_w_scores
plot_df_images_w_scores(index_iterator_to_plot,unnested_images_df,
                        images_folder_path,images_per_row)
plt.suptitle('directly plotting data by plot_df_images_w_scores')
#%% checking image sizes
image_num_to_sample=5
# end of inputs ---------------------------------------------------------------
logger.info('checking image shapes of %d sampled images'%image_num_to_sample)
sampled_indices_list=random.sample(range(len(unnested_images_df)),image_num_to_sample)
for i in sampled_indices_list:
    df_row=unnested_images_df.iloc[i,:]
    image_filename=df_row['image filename']
    image_array=plt.imread(os.path.join(images_folder_path,image_filename))
    print(f'{image_filename} shape:',image_array.shape)
#%% building a transformed torch dataset of unnested np images and testing it
index_iterator_to_plot=range(20)
images_per_row=4
transform_func=transforms.Compose([Rescale(350),RandomCrop(200),ToTensor()])
image_format='np->torch' # use when using ToTensor() transform
#image_format='np' # use when not using ToTensor() transform
# end of inputs ---------------------------------------------------------------
dataset_w_transform_func=unnested_np_images_dataset(images_df=unnested_images_df,
    images_folder_path=images_folder_path,transform_func=transform_func)

# plotting from dataset
plot_dataset_images_w_scores(index_iterator_to_plot,dataset_w_transform_func,
                             images_per_row,image_format=image_format)
plt.suptitle('plotting data from the np images dataset, 1st time')
# plotting from dataset again, to see random transforms in action
plot_dataset_images_w_scores(index_iterator_to_plot,dataset_w_transform_func,
                             images_per_row,image_format=image_format)
plt.suptitle('plotting data from the np images dataset, 2nd time')

# verifying against directly plotting the images by plot_df_images_w_scores
plot_df_images_w_scores(index_iterator_to_plot,unnested_images_df,
                        images_folder_path,images_per_row)
plt.suptitle('directly plotting data by plot_df_images_w_scores')
#%% building a torch dataset of PIL images with torchvision transforms and testing it
"""torchvision.transforms accept PIL images, and not np images that are 
    created when using skimage as presented in 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial

torchvision transforms: https://pytorch.org/docs/stable/torchvision/transforms.html
"""
index_iterator_to_plot=range(20)
images_per_row=4
transform_func=transforms.Compose([
        transforms.Resize(350),
        transforms.RandomCrop(200),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0,hue=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])
"""torchvision.transforms.ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the 
    range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the 
    range [0.0, 1.0] if the PIL Image belongs to one of the modes 
    (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has 
    dtype = np.uint8
In the other cases, tensors are returned without scaling
source: https://pytorch.org/docs/stable/torchvision/transforms.html
"""
image_format='PIL->torch' # use when using ToTensor() transform
#image_format='np' # use when not using ToTensor() transform
# end of inputs ---------------------------------------------------------------
dataset_w_transform_func=unnested_PIL_images_dataset(images_df=unnested_images_df,
    images_folder_path=images_folder_path,transform_func=transform_func)

# plotting from dataset
plot_dataset_images_w_scores(index_iterator_to_plot,dataset_w_transform_func,
                             images_per_row,image_format=image_format)
plt.suptitle('plotting data from the PIL images dataset, 1st time')
# plotting from dataset again, to see random transforms in action
plot_dataset_images_w_scores(index_iterator_to_plot,dataset_w_transform_func,
                             images_per_row,image_format=image_format)
plt.suptitle('plotting data from the PIL images dataset, 2nd time')

# verifying against directly plotting the images by plot_df_images_w_scores
plot_df_images_w_scores(index_iterator_to_plot,unnested_images_df,
                        images_folder_path,images_per_row)
plt.suptitle('directly plotting data by plot_df_images_w_scores')
#%% building a torch dataloader and testing it with no multi-processing
batch_size=20
num_workers=0
#shuffle=True
shuffle=False
images_per_row=5
# end of inputs ---------------------------------------------------------------
dataloader=DataLoader(dataset_w_transform_func,batch_size=batch_size,
                        shuffle=shuffle,num_workers=num_workers)
samples_batch=next(iter(dataloader))
plt.figure()
columns_num=math.ceil(batch_size/images_per_row)
for i in range(batch_size):
    image_array=samples_batch['image array'][i].numpy().transpose((1,2,0))
    image_score=samples_batch['profile score'][i].numpy()
    profile_index=samples_batch['profile index'][i].numpy()
    
    plt.subplot(columns_num,images_per_row,i+1)
    plt.imshow(image_array)
    plt.title('profile index: %s, score: %d'%(profile_index,image_score))
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
plt.suptitle('plotting a batch from the PIL images dataloader')
#%% building a multi-processing torch dataloader and testing it
batch_size=20
num_workers=4
#shuffle=True
shuffle=False
images_per_row=5
# end of inputs ---------------------------------------------------------------
dataloader=DataLoader(dataset_w_transform_func,batch_size=batch_size,
                        shuffle=shuffle,num_workers=num_workers)
if __name__=='__main__': # required in Windows for multi-processing
    samples_batch=next(iter(dataloader))
    
plt.figure()
columns_num=math.ceil(batch_size/images_per_row)
for i in range(batch_size):
    image_array=samples_batch['image array'][i].numpy().transpose((1,2,0))
    image_score=samples_batch['profile score'][i].numpy()
    profile_index=samples_batch['profile index'][i].numpy()
    
    plt.subplot(columns_num,images_per_row,i+1)
    plt.imshow(image_array)
    plt.title('profile index: %s, score: %d'%(profile_index,image_score))
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
plt.suptitle('plotting a batch from the PIL images dataloader')
#%% splitting to train and val datsets and dataloaders
dataset_to_split=dataset_w_transform_func

validation_ratio=0.3

batch_size=20
workers=0
#workers=4
#shuffle_dataset_indices_for_split=True # dataset indices for dataloaders are shuffled before splitting to train and validation indices
shuffle_dataset_indices_for_split=False
#dataloader_shuffle=True # samples are shuffled inside each dataloader
dataloader_shuffle=False
random_seed=0
#----- end of inputs ----------------------------------------------------------
dataset_length=len(dataset_to_split)
dataset_indices=list(range(dataset_length))
split_index=int((1-validation_ratio)*dataset_length)
if shuffle_dataset_indices_for_split:
    np.random.seed(random_seed)
    np.random.shuffle(dataset_indices)
train_indices=dataset_indices[:split_index]
val_indices=dataset_indices[split_index:]

# splitting the dataset to train and val
train_dataset=torch.utils.data.Subset(dataset_to_split,train_indices)
val_dataset=torch.utils.data.Subset(dataset_to_split,val_indices)

# creating the train and val dataloaders
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,
                        num_workers=workers,shuffle=dataloader_shuffle)
val_dataloader=DataLoader(val_dataset,batch_size=batch_size,
                        num_workers=workers,shuffle=dataloader_shuffle)
dataloaders={'train':train_dataloader,'val':val_dataloader}
dataloader_lengths={'train':len(train_indices),'val':len(val_indices)}
#%% verifying the split in dataset
first_sample=dataset_to_split[train_indices[0]]
first_train_sample=train_dataset[0]

split_sample=dataset_to_split[val_indices[0]]
first_val_sample=val_dataset[0]
print('check that first_sample=~first_train_sample and split_sample=~first_val_sample (up to random transforms in dataset def)')
#%% verifying the train and val dataloaders
if __name__=='__main__' or workers==0: # required in Windows for multi-processing
    samples_batches={'train':next(iter(dataloaders['train'])),
                     'val':next(iter(dataloaders['val']))}

columns_num=math.ceil(batch_size/images_per_row)
for phase in ['train','val']:
    samples_batch=samples_batches[phase]
    plt.figure()
    for i in range(batch_size):
        image_array=samples_batch['image array'][i].numpy().transpose((1,2,0))
        image_score=samples_batch['profile score'][i].numpy()
        profile_index=samples_batch['profile index'][i].numpy()
        
        plt.subplot(columns_num,images_per_row,i+1)
        plt.imshow(image_array)
        plt.title('profile index: %s, score: %d'%(profile_index,image_score))
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    plt.suptitle('plotting a batch from the PIL images %s dataloader'%phase)
#%% verifying the split in dataloaders
if __name__ == '__main__' or workers==0:
    first_train_batch=next(iter(dataloaders['train']))
    first_val_batch=next(iter(dataloaders['val']))
    for phase in ['train','val']:
        if dataloader_shuffle:
            internal_length=len(dataloaders[phase].sampler.indices)
        else:
            internal_length=len(dataloaders[phase].sampler.data_source)
        logger.info('%s internal dataloader length == dataloader_lengths[%s]: %d'%(
                phase,phase,internal_length==dataloader_lengths[phase]))
    
    if dataloader_shuffle or shuffle_dataset_indices_for_split:
        logger.info('to verify the splitting, execute with shuffle_dataset_indices_for_split=False and dataloader_shuffle=False')
    else:
        split_sample=dataset_to_split[split_index]
        if len(train_indices)%batch_size==0:
            logger.info('verify that the first sample in first_train_batch is split_sample=dataset_to_split[split_index]')
        else:
            last_train_batch=dataloaders['train'].dataset[-1]
            logger.info('verify that the last sample in last_train_batch is split_sample=dataset_to_split[split_index]')
