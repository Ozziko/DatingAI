# -*- coding: utf-8 -*-
"""@author: oz.livneh@gmail.com
This script informs of profiles in profiles_df that refer to images that do 
    not exist in images_folder_path, and offers to delete those profiles from 
    profiles_df
"""

#%% inputs
data_folder_path=r'D:\AI Data\DatingAI\Data'
images_folder_path=r'D:\AI Data\DatingAI\Data\Images'
df_pickle_file_name='profiles_df.pickle'

#%% imports
import logging
logging.basicConfig(format='%(asctime)s (%(levelname)s): %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger('sentence embedding training')
logger.setLevel(logging.INFO)
import os
import pandas as pd

#%% comparing and merging
file_path=os.path.join(data_folder_path,df_pickle_file_name)
profiles_df=pd.read_pickle(os.path.join(images_folder_path,file_path))
profile_rows_to_delete=[]
profiles_df.index=range(len(profiles_df))
for row_index in range(len(profiles_df)):
    row_series=profiles_df.iloc[row_index]
    for filename in row_series['image filenames']:
        if not os.path.isfile(os.path.join(images_folder_path,filename)):
            logging.warning(f'profile {row_index}: {filename} not found in {images_folder_path}!')
            profile_rows_to_delete.append(row_index)
profile_rows_to_delete=set(profile_rows_to_delete)
n_to_delete=len(profile_rows_to_delete)

if n_to_delete>0:
    profiles_df_dropped=profiles_df.drop(profile_rows_to_delete,axis=0)
    profiles_df_dropped.index=range(len(profiles_df_dropped))
else:
    logger.info(f'no profiles to delete, all images in profiles_df exist in {images_folder_path}')

#%% saving
if n_to_delete>0:
    del_approval=input(f'do you approve deleting {n_to_delete} profiles with missing image files (overwriting profiles_df with profiles_df_dropped)? y/[n] ')
    if del_approval=='y':
        profiles_df_dropped.to_pickle(file_path)
        logger.info('%s saved'%file_path)
    


