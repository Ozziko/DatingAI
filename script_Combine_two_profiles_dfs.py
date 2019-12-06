# -*- coding: utf-8 -*-
"""@author: oz.livneh@gmail.com
This script combines data from two different sources, assuming they were both 
    scored by the same person! the data of profiles_df_second_path is concatenated 
    to the data of profiles_df_base_path, if the profile in the second df does not 
    already appear on the base df.
    
The tricky part - two different profiles in OKCupid can have different ids but 
    represent the same person (same data and photos)! 
    Therefore, this script adds profiles from the second source only if they 
    do not appear with the same id in the base source (mode='shallow'), and 
    optionally if they do not appear with a different id but the same data and 
    photos (mode='deep')

First move all scraped image files from both origins into a single folder, 
    images_folder_path (over-write/skip duplicates, it's irrelevant)
Set profiles_df_base_path,profiles_df_second_path to both profiles_df.pickle files built by 
    PersonalCupidScraperBot.py and run the script
"""

#%% inputs
profiles_df_base_path=r'D:\My Documents\Dropbox\Python\DatingAI\Data\profiles_df.pickle'
profiles_df_second_path=r'D:\My Documents\Dropbox\Python\DatingAI\Data\sd_profiles_df.pickle'
output_folder_path=r'D:\My Documents\Dropbox\Python\DatingAI\Data'
output_filename='merged_profiles_df.pickle'
#mode='shallow' # adds a profile from profiles_df_second only if it does not exist with the same id in profiles_df_base
mode='deep' # adds a profile from profiles_df_second only if it does not exist in profiles_df_base with the same id, or a different id but the same data and photos
#sort_by_timestamp=False
sort_by_timestamp=True

#%% imports
import logging
logging.basicConfig(format='%(asctime)s (%(levelname)s): %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger('sentence embedding training')
logger.setLevel(logging.INFO)
import os
import pandas as pd

#%% comparing and merging
profiles_df_base=pd.read_pickle(profiles_df_base_path)
profiles_df_second=pd.read_pickle(profiles_df_second_path)
profiles_df_second_rows_to_merge=[]
logger.info('starting to compare dataframes in %s mode. data lengths: (profiles_df_base,profiles_df_second,together)=(%d,%d,%d)'%(
        mode,len(profiles_df_base),len(profiles_df_second),len(profiles_df_base)+len(profiles_df_second)))
profiles_df_base_profile_ids=set(profiles_df_base['profile id'])
for i_row_sec in range(len(profiles_df_second)):
    profile_series=profiles_df_second.iloc[i_row_sec]
    profile_id=profile_series['profile id']
    # checking if the current profile_id from profiles_df_second exists already in profiles_df_base
    if profile_id in profiles_df_base_profile_ids:
        logger.warning('profile_id %s from profiles_df_second (row %d) already exists in profiles_df_base in the same profile id-> not concatenated'%(
                profile_id,i_row_sec))
        continue
    break
    if mode=='deep':
        # checking if the currect profile from profiles_df_second exists in profiles_df_base with a different profile_id
        for i_row_base in range(len(profiles_df_base)):
            base_profile_series=profiles_df_base.iloc[i_row_base,:]
            if (base_profile_series['age']==profile_series['age'] and \
                base_profile_series['location']==profile_series['location'] and \
                base_profile_series['essay dict']==profile_series['essay dict'] and \
                base_profile_series['basic details']==profile_series['basic details'] and \
                base_profile_series['extended details']==profile_series['extended details'] and \
                base_profile_series['image filenames']==profile_series['image filenames']):
                    logger.warning('profile id %s of profiles_df_second (row %d) already exists in profiles_df_base in a different id but the same data -> not concatenated'%(
                            profile_id,i_row_sec))
                    continue
    profiles_df_second_rows_to_merge.append(i_row_sec)

logger.info('completed comparing dataframes, merging')
profiles_df_merged=pd.concat(
        [profiles_df_base,profiles_df_second.iloc[profiles_df_second_rows_to_merge]],
        axis=0,ignore_index=True)
if sort_by_timestamp:
    profiles_df_merged.sort_values(by='timestamp',inplace=True,na_position='first')
logger.info('built profiles_df_merged, whose length is %d',len(profiles_df_merged))

#%% saving
output_file_path=os.path.join(output_folder_path,output_filename)
profiles_df_merged.to_pickle(output_file_path)
logger.info('%s saved'%output_file_path)



