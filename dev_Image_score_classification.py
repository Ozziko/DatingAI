# -*- coding: utf-8 -*-
"""@author: oz.livneh@gmail.com

* All rights of this project and my code are reserved to me, Oz Livneh.
* Feel free to use - for personal use!
* Use at your own risk ;-)

Image score classification: 
    For each image, predicting the score class given to the profile to
    which the image belongs.

Each row in profiles_df (the output of the scraper) represents a profile that has 
    in its 'image filenames' column a nested list of image filenames, 0 or more.
    Therefore the dataset used in the learning in this script is 
    unnested_images_df, created by unnesting the image filenames from profiles_df 
    such as each row in unnested_images_df represents an image, the the columns
    for each row are taken from the columns of the profile to which the image 
    belonged.
    (see #%% reading data, unnesting images from profiles_df (creating unnested_images_df))
    
<<<This script is for R&D, experiments, debugging>>>
This script can be run entirely, or executed by sections (for example in Spyder).
"""

#%% main parameters
#--------- general -------------
#debugging=True # executes the debugging (short) sections, prints results
debugging=False
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
load_all_dataset_to_RAM=False # default; loads images from hard drive for each sample in the batch by the PyTorch efficient (multi-processing) dataloader
#load_all_dataset_to_RAM=True # loads all dataset images to RAM; estimates dataset size and requires user approval

validation_ratio=0.3 # validation dataset ratio from total dataset length

#batch_size_int_or_ratio_float=1e-2 # if float: batch_size=round(batch_size_over_dataset_length*len(dataset_to_split))
batch_size_int_or_ratio_float=8 # if int: this is the batch size
data_workers=0 # 0 means no multiprocessing in dataloaders
#data_workers='cpu cores' # sets data_workers=multiprocessing.cpu_count()

shuffle_dataset_indices_for_split=True # dataset indices for dataloaders are shuffled before splitting to train and validation indices
#shuffle_dataset_indices_for_split=False
dataset_shuffle_random_seed=0 # numpy seed for sampling the indices for the dataset, before splitting to train and val dataloaders
#dataset_shuffle_random_seed=None
dataloader_shuffle=True # samples are shuffled inside each dataloader, on each epoch
#dataloader_shuffle=False

#--------- net -------------
#architecture_is_a_pretrained_model=False
#net_architecture='simple CNN'

architecture_is_a_pretrained_model=True
net_architecture='inception v3'
#net_architecture='resnet18'

#freeze_pretrained_net_weights=True
freeze_pretrained_net_weights=False

loss_name='cross entropy'
yield_probabilities_in_my_models=False # "the standard": net output is a linear layer, not probabilities
#yield_probabilities_in_my_models=True # apply a softmax on the linear output layer to get probabilities from the net!
if yield_probabilities_in_my_models:
    raise RuntimeError("For nn.CrossEntropyLoss: The input is expected to contain raw, unnormalized scores for each class'! To make the net output probabilities, I can use nn.NLLLoss() and nn.LogSoftmax() as the last layer of the net, then transform net outputs to probabilities by exponenting")

#--------- training -------------
train_model_else_load_weights=True
#train_model_else_load_weights=False # instead of training, loads a pre-trained model and uses it
#class_loss_rescaling='none'
#class_loss_rescaling='manual' # rescaling the classes in the loss according to class_rescaling_weights_tensor
#class_rescaled_loss_weights_list=[3,1,2,4,5,3] # class loss rescaling: loss[class]=factor*loss[class], higher factor increases the significance of the class, more predictions and percision of this class and less for other classes
class_loss_rescaling='auto' # rescaling according to 1/(class ground truth distribution in training)

epochs=10
learning_rate=1e-5
momentum=0.9

lr_scheduler_step_size=1
lr_scheduler_decay_factor=0.9

best_model_criterion='max val epoch acc' # criterion for choosing best net weights during training as the final weights
return_to_best_weights_in_the_end=True # when training complets, loads weights of the best net, definied by best_model_criterion
#return_to_best_weights_in_the_end=False

training_progress_ratio_to_log_loss=1 # <=1, inter-epoch logging and reporting loss and metrics during training, period_in_batches_to_log_loss=round(training_progress_ratio_to_log_loss*dataset_samples_number['train']/batch_size)
print_metrics_after_each_epoch=True
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
from collections import Counter
import multiprocessing
if data_workers=='cpu cores':
    data_workers=multiprocessing.cpu_count()

import torch
torch.manual_seed(torch_manual_seed)

import torchvision
import torch.nn as nn
import torch.nn.functional as F


def plot_df_images_w_scores(sample_indices_to_plot,df,images_folder_path,
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
    columns_num=math.ceil(len(sample_indices_to_plot)/images_per_row)
    for i,sample_index in enumerate(sample_indices_to_plot):
        df_row=df.iloc[sample_index,:]
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

class unnested_PIL_images_dataset(torch.utils.data.Dataset):
    def __init__(self,images_df,images_folder_path,transform_func=None):
        self.images_df=images_df
        self.images_folder_path=images_folder_path
        self.transform_func=transform_func

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,idx):
        profile_score=self.images_df['score'].iloc[idx]
        profile_score_code=self.images_df['score code'].iloc[idx]
        profile_index=self.images_df['profile index'].iloc[idx]
        profile_id=self.images_df['profile id'].iloc[idx]
        image_filename=self.images_df['image filename'].iloc[idx]
        
        image_path=os.path.join(self.images_folder_path,image_filename)
        image_array=PIL.Image.open(image_path)
        
        if self.transform_func!=None:
            image_array=self.transform_func(image_array)
        
        sample={'image array':image_array,'profile score':profile_score,
                'profile score code':profile_score_code,'profile index':profile_index,
                'image filename':image_filename,'profile id':profile_id}
        return sample

def build_images_dict_in_RAM(images_df,images_folder_path):
    images_dict={}
    for i in range(len(images_df)):
        image_path=os.path.join(images_folder_path,
                                images_df['image filename'].iloc[i])
        image_array=PIL.Image.open(image_path)
        image_array.load()
        images_dict.update({i:image_array})
    return images_dict

class unnested_PIL_images_dataset_in_RAM(torch.utils.data.Dataset):
    def __init__(self,images_df,images_dict,transform_func=None):
        self.images_df=images_df
        self.images_dict=images_dict
        self.transform_func=transform_func

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,idx):
        profile_score=self.images_df['score'].iloc[idx]
        profile_score_code=self.images_df['score code'].iloc[idx]
        profile_index=self.images_df['profile index'].iloc[idx]
        profile_id=self.images_df['profile id'].iloc[idx]
        image_filename=self.images_df['image filename'].iloc[idx]
        
        image_array=self.images_dict[idx]
        if self.transform_func!=None:
            image_array=self.transform_func(image_array)
        
        sample={'image array':image_array,'profile score':profile_score,
                'profile score code':profile_score_code,'profile index':profile_index,
                'image filename':image_filename,'profile id':profile_id}
        return sample

def plot_dataset_images_w_scores(sample_indices_to_plot,dataset,images_per_row,
                                 image_format='PIL->torch'):
    plt.figure()
    columns_num=math.ceil(len(sample_indices_to_plot)/images_per_row)
    for i,sample_index in enumerate(sample_indices_to_plot):
        sampe=dataset[sample_index]
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

def training_stats_plot(stats_dict,fig,loss_subplot,acc_subplot,
                        plot_loss_in_log_scale=False):
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
        
    acc_subplot.clear() # clearing plot before plotting, to avoid over-plotting
    if len(running_stats_df)>0:
        acc_subplot.plot(100*running_stats_df['accuracy'],'-x',label='running train')
    acc_subplot.plot(100*epoch_train_stats_df['accuracy'],'k-o',label='epoch train')
    acc_subplot.plot(100*epoch_val_stats_df['accuracy'],'r-o',label='epoch val')
    acc_subplot.set_ylabel('accuracy (%)')
    acc_subplot.set_xlabel('epoch')
    acc_subplot.grid()
    acc_subplot.legend(loc='best')
    if plot_loss_in_log_scale:
        acc_subplot.set_yscale('log')
    fig.canvas.draw()

class remainder_time:
    def __init__(self,time_seconds):
        self.time_seconds=time_seconds
        self.hours=int(time_seconds/3600)
        self.remainder_minutes=int((time_seconds-self.hours*3600)/60)
        self.remainder_seconds=time_seconds-self.hours*3600-self.remainder_minutes*60

logger.info('script initialized')

#%% reading data, unnesting images from profiles_df (creating unnested_images_df)
"""
explanation about the unnesting of profiles_df to unnested_images_df appears in 
    the script documentation in the beginning of the script.

'profile index' in unnested_images_df is the row index in profiles_df
"""
profiles_df_path=os.path.join(data_folder_path,df_pickle_file_name)
profiles_df=pd.read_pickle(profiles_df_path)

# finding score column (column name in the format of 'score (levels=%d)'%score_levels)
score_column_name=None
for column in profiles_df.columns:
    if 'score' in column:
        score_column_name=column
        break
if score_column_name==None:
    raise RuntimeError("no existing column name in profiles_df contains 'score'!")

# translating scores to score codes
scores_set=set(profiles_df[score_column_name])
score_to_code_dict={}
code_to_score_dict={}
for code,score in enumerate(sorted(scores_set)):
    score_to_code_dict.update({score:code})
    code_to_score_dict.update({code:score})

# unnesting images
unnested_images_dict={} # = {filename:{'score':...,'score code':...,'profile id':...}}
unnested_images_df_list=[] # = ['profile index','profile id','score','score code','image filename']
for row_index in range(len(profiles_df)):
    row_series=profiles_df.iloc[row_index,:]
    profile_id=row_series['profile id']
    profile_score=row_series[score_column_name]
    profile_score_code=score_to_code_dict[profile_score]
    for filename in row_series['image filenames']:
        if filename=='pq_400.pn': # skipping the blank profile image of a strange format
            continue
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

if max_dataset_length>0 and max_dataset_length<len(unnested_images_df):
    user_data_approval=input('ATTENTION: downsampling is chosen - building a dataset by sampling only max_dataset_length=%d samples from all available data! approve? y/[n] '%(round(max_dataset_length)))
    if user_data_approval!='y':
        raise RuntimeError('user did not approve dataset max_dataset_length sampling!')
    random.seed(seed_for_dataset_downsampling)
    sampled_indices=random.sample(range(len(unnested_images_df)),max_dataset_length)
    unnested_images_df=unnested_images_df.iloc[sampled_indices]

logger.info('comleted unnesting images from profiles_df to unnested_images_df of length %.1e'%(len(unnested_images_df)))

#%% (debugging) checking image sizes
image_num_to_sample=5
# end of inputs ---------------------------------------------------------------
if debugging:
    logger.info('checking image shapes of %d sampled images'%image_num_to_sample)
    sampled_indices_list=random.sample(range(len(unnested_images_df)),image_num_to_sample)
    for i in sampled_indices_list:
        df_row=unnested_images_df.iloc[i,:]
        image_filename=df_row['image filename']
        image_array=plt.imread(os.path.join(images_folder_path,image_filename))
        print(f'{image_filename} shape:',image_array.shape)

#%% (debugging) checking image reading
if debugging:
    image_path=os.path.join(images_folder_path,unnested_images_df['image filename'].iloc[0])
    image_array=PIL.Image.open(image_path)
    #image_array.load()
    image_array
    np.array(image_array)
    #image_array.close()

#%% building a torch dataset of PIL images with torchvision transforms
"""torchvision.transforms accept PIL images, and not np images that are 
    created when using skimage as presented in 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial

torchvision transforms: https://pytorch.org/docs/stable/torchvision/transforms.html
"""
if not random_transforms:
    input('random_transforms=False was set, meaning no data augmentation, aknowledge by hitting')
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
n_to_sample_for_data_size_estimation=10
# end of inputs ---------------------------------------------------------------

dataset_w_transform=unnested_PIL_images_dataset(images_df=unnested_images_df,
    images_folder_path=images_folder_path,transform_func=transform_func)

if load_all_dataset_to_RAM:
    sampled_sample_indices=random.sample(range(len(dataset_w_transform)),n_to_sample_for_data_size_estimation)
    sampled_images_dict_in_RAM=build_images_dict_in_RAM(unnested_images_df.iloc[sampled_sample_indices],images_folder_path)
    expected_sampled_images_dict_in_RAM_size_MB=sys.getsizeof(
            sampled_images_dict_in_RAM)/1e6/n_to_sample_for_data_size_estimation*len(dataset_w_transform)
    user_decision_RAM=input('load_all_dataset_to_RAM=True was set, estimated dataset size based on %d random samples: %.1eMB, continue? y/[n] '%(
            n_to_sample_for_data_size_estimation,expected_sampled_images_dict_in_RAM_size_MB))
    if user_decision_RAM=='y':
        logger.info('starting to load all dataset to RAM')
        images_dict_in_RAM=build_images_dict_in_RAM(unnested_images_df,images_folder_path)
        dataset_w_transform=unnested_PIL_images_dataset_in_RAM(images_df=unnested_images_df,
                            images_dict=images_dict_in_RAM,transform_func=transform_func)
        logger.info('completed loading all dataset to RAM, size: %.1eMB'%(sys.getsizeof(images_dict_in_RAM)/1e6))
    else:
        logger.info('user disapproved loading all dataset to RAM, keeping it on the hard drive and loading with a dataloader')

sample_size=dataset_w_transform[0]['image array'].size()
sample_pixels_per_channel=sample_size[1]*sample_size[2]
sample_pixels_all_channels=sample_size[0]*sample_pixels_per_channel
logger.info('set a PyTorch dataset of length %.2e, input size (assuming it is constant): (%d,%d,%d)'%(
        len(unnested_images_df),sample_size[0],sample_size[1],sample_size[2]))

#%% (debugging) verifying dataset by plotting
#sample_indices_to_plot=range(20) # for dataset plotting verification
random.seed(0)
sample_indices_to_plot=random.sample(range(len(unnested_images_df)),20)
images_per_row=4
# end of inputs ---------------------------------------------------------------

if debugging:
    if architecture_is_a_pretrained_model:
        input('since architecture_is_a_pretrained_model=True, images are normalized (strangely) and PIL will raise warnings, approve by hitting anything')
    
    plot_dataset_images_w_scores(sample_indices_to_plot,dataset_w_transform,
                                 images_per_row,image_format='PIL->torch')
    plt.suptitle('plotting from pytorch dataset, 1st time')
    
    
    plot_dataset_images_w_scores(sample_indices_to_plot,dataset_w_transform,
                                 images_per_row,image_format='PIL->torch')
    plt.suptitle('plotting from pytorch dataset, 2st time (to see random transforms)')
    
    plot_df_images_w_scores(sample_indices_to_plot,unnested_images_df,
                            images_folder_path,images_per_row)
    plt.suptitle('plotting from unnested_images_df')

#%% splitting to train and val datsets and dataloaders
dataset_to_split=dataset_w_transform

if isinstance(batch_size_int_or_ratio_float,int):
    batch_size=batch_size_int_or_ratio_float
elif isinstance(batch_size_int_or_ratio_float,float):
    batch_size=round(batch_size_int_or_ratio_float*len(dataset_to_split))
else:
    raise RuntimeError('unsupported batch_size input!')
if batch_size<1:
    batch_size=1
    logger.warning('batch_size=round(batch_size_over_dataset_length*len(dataset_to_split))<1 so batch_size=1 was set')
if batch_size==1:
    user_batch_size=input('batch_size=1 should cause errors since batch_size>1 is generally assumed! enter a new batch size equal or larger than 1, or smaller than 1 to abort: ')
    if user_batch_size<1:
        raise RuntimeError('aborted by user batch size decision')
    else:
        batch_size=round(user_batch_size)

dataset_length=len(dataset_to_split)
dataset_indices=list(range(dataset_length))
split_index=int((1-validation_ratio)*dataset_length)
if shuffle_dataset_indices_for_split:
    np.random.seed(dataset_shuffle_random_seed)
    np.random.shuffle(dataset_indices)
train_indices=dataset_indices[:split_index]
val_indices=dataset_indices[split_index:]

# splitting the dataset to train and val
train_dataset=torch.utils.data.Subset(dataset_to_split,train_indices)
val_dataset=torch.utils.data.Subset(dataset_to_split,val_indices)

# creating the train and val dataloaders
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                        num_workers=data_workers,shuffle=dataloader_shuffle)
val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,
                        num_workers=data_workers,shuffle=dataloader_shuffle)

# structuring
dataset_indices={'train':train_indices,'val':val_indices}
datasets={'train':train_dataset,'val':val_dataset}
dataset_samples_number={'train':len(train_dataset),'val':len(val_dataset)}

dataloaders={'train':train_dataloader,'val':val_dataloader}
dataloader_batches_number={'train':len(train_dataloader),'val':len(val_dataloader)}

logger.info('dataset split to training and validation datasets and dataloaders with validation_ratio=%.1f, lengths: (train,val)=(%d,%d)'%(
        validation_ratio,dataset_samples_number['train'],dataset_samples_number['val']))

# statistics
#ground_truth_counters={phase:Counter([sample['profile score code'] \
#                                      for sample in datasets[phase]]) \
#            for phase in ('train','val')}
# avoiding this computation that scans all and loads images on-the-fly just to read their labels, using a much more efficient computation:
ground_truth_counters={phase:Counter(unnested_images_df['score code'].iloc[dataset_indices[phase]].tolist())
            for phase in ('train','val')}
class_distributions_dict={phase:{('%d (%%)'%(code_to_score_dict[class_code])):\
                            100*ground_truth_counters[phase][class_code]/dataset_samples_number[phase]\
                            for class_code in code_to_score_dict}
            for phase in ('train','val')}
class_distributions_df=pd.DataFrame.from_dict(class_distributions_dict,orient='index')
print('class ground trouth distributions:\n',class_distributions_df)

if class_loss_rescaling=='auto':
    class_rescaled_loss_weights_list=[]
    for class_code in range(len(score_to_code_dict)):
        if ground_truth_counters['train'][class_code]!=0:
            class_rescaled_loss_weight=dataset_samples_number['train']/ground_truth_counters['train'][class_code]
        else:
            class_rescaled_loss_weight=1
            logger.warning('class %s has no occurances in the data (its rescaled loss weight is 1)'%(code_to_score_dict[class_code]))
        class_rescaled_loss_weights_list.append(class_rescaled_loss_weight)

#%% (debugging) verifying dataloaders
images_per_row=4
# end of inputs ---------------------------------------------------------------

if debugging:
    if __name__=='__main__' or data_workers==0: # required in Windows for multi-processing
        samples_batches={}
        for phase in ['train','val']:
            samples_batch=next(iter(dataloaders[phase]))
            samples_batches.update({phase:samples_batch})
    else:
        raise RuntimeError('cannot use multiprocessing (data_workers>0 in dataloaders) in Windows when executed not as main!')
        
    columns_num=math.ceil(batch_size/images_per_row)
    for phase in ['train','val']:
        plt.figure()
        for i in range(batch_size):
            samples_batch=samples_batches[phase]
            image_array=samples_batch['image array'][i].numpy().transpose((1,2,0))
            image_score=samples_batch['profile score'][i].numpy()
            profile_index=samples_batch['profile index'][i].numpy()
            
            plt.subplot(columns_num,images_per_row,i+1)
            plt.imshow(image_array)
            plt.title('profile index: %s, score: %d'%(profile_index,image_score))
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
        plt.suptitle('plotting a batch from the %s dataloader'%phase)

#%% setting the NN
if training_progress_ratio_to_log_loss>1:
    raise RuntimeError('invalid training_progress_ratio_to_log_loss=%.2f, must be <=1'%training_progress_ratio_to_log_loss)
period_in_batches_to_log_loss=round(training_progress_ratio_to_log_loss*dataset_samples_number['train']/batch_size) # logging only during training
class_rescaling_weights_tensor=torch.tensor(class_rescaled_loss_weights_list).float()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if plot_realtime_stats_on_logging or plot_realtime_stats_after_each_epoch:
    logger.warning('plotting from inside the net loop is not working, should be debugged...')

if net_architecture=='simple CNN':
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3,6,5)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
            self.fc1 = nn.Linear(141376,120)
            self.fc2 = nn.Linear(120,84)
            self.fc3 = nn.Linear(84,len(scores_set))
        def forward(self,x):
            x=F.relu(self.conv1(x))
            x=self.pool(x)
            x=F.relu(self.conv2(x))
            x=self.pool(x)            
            x=x.view(-1,np.array(x.shape[1:]).prod()) # don't use x.view(batch_size,-1), which fails for batches smaller than batch_size (at the end of the dataloader)
            x=F.relu(self.fc1(x))
            x=F.relu(self.fc2(x))
            if yield_probabilities_in_my_models:
                x=F.softmax(self.fc3(x),dim=1)
            else:
                x=self.fc3(x)
            return x
    model=Net()
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
elif net_architecture=='resnet18':
    model=torchvision.models.resnet18(pretrained=True)
    if freeze_pretrained_net_weights:
        for param in model.parameters():
            param.requires_grad=False
        parameters_to_optimize=model.fc.parameters()
    else:
        parameters_to_optimize=model.parameters()
    model.fc=nn.Linear(model.fc.in_features,len(scores_set)) # Parameters of newly constructed modules have requires_grad=True by default
    optimizer=torch.optim.SGD(parameters_to_optimize,lr=learning_rate,momentum=momentum)
elif net_architecture=='inception v3':
    model=torchvision.models.inception_v3(pretrained=True)
    
    if freeze_pretrained_net_weights:
        for param in model.parameters():
            param.requires_grad=False
    # Parameters of newly constructed modules have requires_grad=True by default:
    model.AuxLogits.fc=nn.Linear(768,len(scores_set))
    model.fc=nn.Linear(2048,len(scores_set))
    
    if freeze_pretrained_net_weights:
        parameters_to_optimize=[]
        for name,parameter in model.named_parameters():
            if parameter.requires_grad:
                parameters_to_optimize.append(parameter)
    else:
        parameters_to_optimize=model.parameters()
    optimizer=torch.optim.SGD(parameters_to_optimize,lr=learning_rate,momentum=momentum)
else:
    raise RuntimeError('untreated net_architecture!')

model=model.to(device)
if loss_name=='cross entropy':
    if class_loss_rescaling:
        loss_fn=nn.CrossEntropyLoss(weight=class_rescaling_weights_tensor).to(device)
    else:
        loss_fn=nn.CrossEntropyLoss().to(device)
else:
    raise RuntimeError('untreated loss_name input')
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,
    step_size=lr_scheduler_step_size,gamma=lr_scheduler_decay_factor)

#%% (debugging) verifying the model outputs (comment last lines from Net.forward() to check outputs of earlier lines)
if debugging:
    if __name__=='__main__' or data_workers==0:
        batch=next(iter(dataloaders['train']))
        labels=batch['profile score'].long()
        labels_list=labels.tolist()
        labels=labels.to(device)
        
        images=batch['image array']
        images=images.to(device)
        print('images shape:',images.shape)
#        if net_architecture=='inception v3' and phase=='train':
#            outputs,aux_outputs=model(images)
#        else:
#            outputs=model(images)
        model.eval()
        outputs=model(images)
        print('outputs shape:',outputs.shape)
    else:
        raise RuntimeError('cannot use multiprocessing (data_workers>0 in dataloaders) in Windows when executed not as main!')

#%% training the net
if train_model_else_load_weights and (__name__=='__main__' or data_workers==0):
    stats_dict={'train':{'epoch metrics':{},
                     'running metrics':{}}, # running = since last log
                     'val':{'epoch metrics':{}}}
    
    total_batches=epochs*(dataloader_batches_number['train']+dataloader_batches_number['val'])
    
    pytorch_total_wts=sum(p.numel() for p in model.parameters())
    pytorch_trainable_wts=sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("started training '%s' net on %s, trainable/total weigths: %d/%d"%(
        net_architecture,device,pytorch_trainable_wts,pytorch_total_wts))
    tic=time()
    for epoch in range(epochs):
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluate mode
            
            loss_since_last_log=0.0 # must be a float
            true_predictions_since_last_log=0
            samples_processed_since_last_log=0
            epoch_loss=0.0 # must be a float
            class_predicted_positives_dict={key:0 for key in code_to_score_dict}
            class_true_positives_dict={key:0 for key in code_to_score_dict}
            
            for i_batch,batch in enumerate(dataloaders[phase]):
                images=batch['image array']
                images=images.to(device)
                
                labels=batch['profile score code'].long()
                labels_list=labels.tolist()
                labels=labels.to(device)
                
                optimizer.zero_grad() # zero the parameter gradients
                
                # forward
                with torch.set_grad_enabled(phase=='train'): # if phase=='train' it tracks tensor history for grad calc
                    if net_architecture=='inception v3' and phase=='train':
                        outputs,aux_outputs=model(images)
                        loss1=loss_fn(outputs,labels)
                        loss2=loss_fn(aux_outputs,labels)
                        loss=loss1+0.4*loss2 # in train mode it has an auxiliary output (to deal with gradient decay); see https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
                    else:
                        outputs=model(images)
                        loss=loss_fn(outputs,labels)
                    if torch.isnan(loss):
                        raise RuntimeError('reached NaN loss - aborting training!')
                    _,predicted_classes=torch.max(outputs,dim=1)
                    # backward + optimize if training
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                
                # accumulating stats
                predicted_classes_list=predicted_classes.tolist()
                for i,prediction in enumerate(predicted_classes_list):
                    class_predicted_positives_dict[prediction]+=1
                    if prediction==labels_list[i]:
                        class_true_positives_dict[prediction]+=1
            
                samples_number=len(labels)
                samples_processed_since_last_log+=samples_number
                current_loss=loss.item()*samples_number # the loss is averaged across samples in each minibatch, so it is multiplied to return to a total
                current_true_predictions=torch.sum(predicted_classes==labels).item()
                
                loss_since_last_log+=current_loss
                true_predictions_since_last_log+=current_true_predictions
                
                epoch_loss+=current_loss
                
                if phase=='train' and i_batch%period_in_batches_to_log_loss==(period_in_batches_to_log_loss-1):
                    loss_since_last_log_per_sample=loss_since_last_log/samples_processed_since_last_log
                    acc_since_last_log=true_predictions_since_last_log/samples_processed_since_last_log
            
                    completed_batches=epoch*(dataloader_batches_number['train']+dataloader_batches_number['val'])+(i_batch+1)
                    completed_batches_progress=completed_batches/total_batches
                    passed_seconds=time()-tic
                    expected_seconds=passed_seconds/completed_batches_progress*(1-completed_batches_progress)
                    expected_remainder_time=remainder_time(expected_seconds)
                    
                    logger.info('(epoch %d/%d, batch %d/%d, %s) running loss per sample (since last log): %.3e, running accuracy (since last log): %.3f%%\n\tETA: %dh:%dm:%.0fs'%(
                                epoch+1,epochs,i_batch+1,dataloader_batches_number[phase],phase,
                                loss_since_last_log_per_sample,
                                100*acc_since_last_log,
                                expected_remainder_time.hours,expected_remainder_time.remainder_minutes,expected_remainder_time.remainder_seconds))
                    
                    partial_epoch=epoch+completed_batches_progress
                    stats_dict[phase]['running metrics'].update({partial_epoch:
                        {'batch':i_batch+1,
                         'loss per sample':loss_since_last_log_per_sample,
                         'accuracy':acc_since_last_log}})
    
                    loss_since_last_log=0.0 # must be a float
                    true_predictions_since_last_log=0
                    samples_processed_since_last_log=0
            
            # epoch stats
            epoch_loss_per_sample=epoch_loss/dataset_samples_number[phase]
            epoch_acc=sum(class_true_positives_dict.values())/dataset_samples_number[phase]
            class_metrics_dict={}
            for class_code in code_to_score_dict:
                if class_predicted_positives_dict[class_code]==0:
                    class_precision=np.nan
                else:
                    class_precision=class_true_positives_dict[class_code]/class_predicted_positives_dict[class_code]
                if ground_truth_counters[phase][class_code]==0:
                    class_recall=np.nan
                else:
                    class_recall=class_true_positives_dict[class_code]/ground_truth_counters[phase][class_code]
                
                class_metrics_dict.update({code_to_score_dict[class_code]:{
                    'known pos/tot (%)':100*ground_truth_counters[phase][class_code]/dataset_samples_number[phase],
                    'predicted/tot (%)':100*class_predicted_positives_dict[class_code]/dataset_samples_number[phase],
                    'precision (%)':100*class_precision,
                    'recall (%)':100*class_recall}})
            
            stats_dict[phase]['epoch metrics'].update({epoch:
                        {'loss per sample':epoch_loss_per_sample,
                         'accuracy':epoch_acc,'class metrics':class_metrics_dict}})
            if  phase=='val':
                if best_model_criterion=='max val epoch acc':
                    best_criterion_current_value=epoch_acc
                    if epoch==0:
                        best_criterion_best_value=best_criterion_current_value
                        best_model_wts=copy.deepcopy(model.state_dict())
                        best_epoch=epoch
                    else:
                        if best_criterion_current_value>best_criterion_best_value:
                            best_criterion_best_value=best_criterion_current_value
                            best_model_wts=copy.deepcopy(model.state_dict())
                            best_epoch=epoch
                
                completed_epochs_progress=(epoch+1)/epochs
                passed_seconds=time()-tic
                expected_seconds=passed_seconds/completed_epochs_progress*(1-completed_epochs_progress)
                expected_remainder_time=remainder_time(expected_seconds)
                
                # not printing epoch stats for training, since in this phase they are being measured while the weights are being updated, unlike in validation where stats are measured with no update
                logger.info('(epoch %d, %s) epoch loss per sample: %.3e, epoch accuracy: %.3f%%\n\tETA: %dh:%dm:%.0fs'%(
                    epoch+1,
                    phase,
                    epoch_loss_per_sample,
                    100*epoch_acc,
                    expected_remainder_time.hours,
                    expected_remainder_time.remainder_minutes,
                    expected_remainder_time.remainder_seconds))
                
                if print_metrics_after_each_epoch:
                    print('(epoch %d) %s class_metrics_dict:'%(epoch+1,phase))
                    class_metrics_df=pd.DataFrame.from_dict(class_metrics_dict,orient='index')
                    print(class_metrics_df)
                else:
                    print('-'*10)
                
    toc=time()
    elapsed_sec=toc-tic
    pytorch_total_wts=sum(p.numel() for p in model.parameters())
    pytorch_trainable_wts=sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info('finished training %d epochs in %dm:%.1fs'%(
            epochs,elapsed_sec//60,elapsed_sec%60))
    if return_to_best_weights_in_the_end:
        model.load_state_dict(best_model_wts)
        logger.info("loaded weights of best model according to '%s' criterion: best value %.3f achieved in epoch %d"%(
                best_model_criterion,best_criterion_best_value,best_epoch+1))
    if not (plot_realtime_stats_on_logging or plot_realtime_stats_after_each_epoch):
        fig=plt.figure()
        plt.suptitle('model stats')
        loss_subplot=plt.subplot(1,2,1)
        acc_subplot=plt.subplot(1,2,2)
    training_stats_plot(stats_dict,fig,loss_subplot,acc_subplot)
else: # train_model_else_load_weights==False
    ui_model_name=input('model weights file name to load: ')
    model_weights_file_path=os.path.join(models_folder_path,ui_model_name)
    if not os.path.isfile(model_weights_file_path):
        raise RuntimeError('%model_weights_path does not exist!')
    model_weights=torch.load(model_weights_file_path)
    model.load_state_dict(model_weights)
    logger.info('model weights from %s were loaded'%model_weights_file_path)

#%% post-training model evaluation
"""the validation class_metrics_df measured here in the model evaluation must be identical to those measured during the
    last/best epoch, UNLIKE the training metrics - since the train phase metrics measured during training were being 
    measured while the weights were being updated in batches (!), not after the train phase epoch completed (which 
    would require another iteration on the train dataloader to measure metrics, as is done here without training)
"""
logger.info('started model evaluation')
evaluation_metrics_dict={}
model.eval() # set model to evaluate mode
for phase in ['train','val']:
    epoch_loss=0.0 # must be a float
    
    class_predicted_positives_dict={key:0 for key in code_to_score_dict}
    class_true_positives_dict={key:0 for key in code_to_score_dict}
    false_predictions={}
    
    for i_batch,batch in enumerate(dataloaders[phase]):
        images=batch['image array']
        images=images.to(device)
                
        labels=batch['profile score code'].long()
        labels_list=labels.tolist()
        labels=labels.to(device)
                
        # forward
        with torch.set_grad_enabled(False):
            outputs=model(images)
            loss=loss_fn(outputs,labels)    
            _,predicted_classes=torch.max(outputs,dim=1)
        
        # accumulating stats
        predicted_classes_list=predicted_classes.tolist()
        for i,prediction in enumerate(predicted_classes_list):
            class_predicted_positives_dict[prediction]+=1
            if prediction==labels_list[i]:
                class_true_positives_dict[prediction]+=1
            else:
                false_predictions.update({len(false_predictions):{
                        'profile id':batch['profile id'][i],
                        'image filename':batch['image filename'][i],
                        'prediction':prediction,
                        'label':labels_list[i]}})
    
        samples_number=len(labels)
        current_loss=loss.item()*samples_number # the loss is averaged across samples in each minibatch, so it is multiplied to return to a total
        current_true_predictions=torch.sum(predicted_classes==labels).item()
        
        epoch_loss+=current_loss

    # epoch stats
    epoch_loss_per_sample=epoch_loss/dataset_samples_number[phase]
    epoch_acc=sum(class_true_positives_dict.values())/dataset_samples_number[phase]
    class_metrics_dict={}
    for class_code in code_to_score_dict:
        if class_predicted_positives_dict[class_code]==0:
            class_precision=np.nan
        else:
            class_precision=class_true_positives_dict[class_code]/class_predicted_positives_dict[class_code]
        if ground_truth_counters[phase][class_code]==0:
            class_recall=np.nan
        else:
            class_recall=class_true_positives_dict[class_code]/ground_truth_counters[phase][class_code]
        
        class_metrics_dict.update({code_to_score_dict[class_code]:{
            'known pos/tot (%)':100*ground_truth_counters[phase][class_code]/dataset_samples_number[phase],
            'predicted/tot (%)':100*class_predicted_positives_dict[class_code]/dataset_samples_number[phase],
            'precision (%)':100*class_precision,
            'recall (%)':100*class_recall}})
    evaluation_metrics_dict.update({phase:{
            'class metrics':class_metrics_dict,
            'false predictions':false_predictions}})
    logger.info('(post-training,  %s) loss per sample: %.3e, total accuracy: %.3f%%'%(
                        phase,epoch_loss_per_sample,100*epoch_acc))
    print('(post-training) %s class_metrics_dict:'%(phase))
    class_metrics_df=pd.DataFrame.from_dict(class_metrics_dict,orient='index')
    print(class_metrics_df)
logger.info('completed model evaluation')
#%% analysis of false predictions - confusion matrix
phase_to_analyze='val'
#----- end of inputs ----------------------------------------------------------

false_predictions_df=pd.DataFrame.from_dict(
        evaluation_metrics_dict[phase_to_analyze]['false predictions'],
        orient='index')
class_false_predictions_counter=Counter(false_predictions_df['prediction'])
class_false_predictions_labels_dist={}
for false_prediction in class_false_predictions_counter:
    class_false_predictions_df=false_predictions_df[false_predictions_df['prediction']==false_prediction]
    class_false_predictions_labels_counter=Counter(class_false_predictions_df['label'])
    class_false_predictions_labels_counts_normalized={}
#    class_false_predictions_labels_dist.update({code_to_score_dict[false_prediction]:
#        class_false_predictions_labels_counter})
    for label,count in class_false_predictions_labels_counter.items():
        if class_false_predictions_counter[false_prediction]==0:
            counts_normalized=None
        else:
            counts_normalized=count/class_false_predictions_counter[false_prediction]
        class_false_predictions_labels_counts_normalized.update({'true %d'%code_to_score_dict[label]:counts_normalized})
    class_false_predictions_labels_dist.update({'false %d'%code_to_score_dict[false_prediction]:
        class_false_predictions_labels_counts_normalized})
#print('distribution (%) of false predictions:\n',class_false_predictions_labels_dist)
class_false_predictions_labels_df=pd.DataFrame.from_dict(class_false_predictions_labels_dist,
        orient='index')
print('confusion matrix:\n',class_false_predictions_labels_df)
#%% saving the model
if offer_mode_saving and train_model_else_load_weights:
    try: os.mkdir(models_folder_path)
    except FileExistsError: pass # if the folder exists already - do nothing
    
    saving_decision=input('save model weights? [y]/n ')
    if saving_decision!='n':
        ui_model_name=input('name model weights file: ')
        model_weights_file_path=os.path.join(models_folder_path,ui_model_name+'.ptweights')
        if os.path.isfile(model_weights_file_path):
            alternative_filename=input('%s already exists, give a different file name to save, the same file name to over-write, or hit enter to abort: '%model_weights_file_path)
            if alternative_filename=='':
                raise RuntimeError('aborted by user')
            else:
                model_weights_file_path=os.path.join(models_folder_path,alternative_filename+'.ptweights')
        torch.save(model.state_dict(),model_weights_file_path)       
        logger.info('%s saved'%model_weights_file_path)
#%%
logger.info('script completed')
