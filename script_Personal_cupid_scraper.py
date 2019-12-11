# -*- coding: utf-8 -*-
"""@author: oz.livneh@gmail.com

* All rights of this project and my code are reserved to me, Oz Livneh.
* Feel free to use - for personal use!
* Use at your own risk ;-)

This script is a Personal Cupid Scraper: opens a Chrome browser 
    (selenium-driven) for a user to login into his/her personal OK Cupid 
    account, scrapes all data (textual+images) from the profiles 
    suggested in DoubleTakem, then for each profile it asks the user 
    to give a score and accordingly likes/passes the profile and 
    advances to the next profile.
    
All data – text fields, images and the user scores – is saved locally (!) on 
    the disk (path stated in data_folder_path parameter) in a pandas DataFrame 
    (profiles_df) AND images (downloaded to a folder according 
    to images_folder_path parameter).

Scoring system:
    score_levels parameter is the number of the (integer!) score levels in 
    each direction - positive (+, Like) and negative (-, Pass), 
    which are also translated to actions for the site - Like or Pass on the 
    profile.
    
    For example, score_levels=1 creates a score dictionary of 
    {-1: '-', 1: '+'}, only binary Like/Pass, which is too simple and a waste 
    of details in my opinion, since a user can usually be more specific about 
    the strength of Like/Pass.
    
    Therefore the default value I chose is score_levels=3, creating a score 
    dictionary of {-3: '---', -2: '--', -1: '-', 1: '+', 2: '++', 3: '+++'}, 
    giving more levels of strength in each direction. Notice that the score 
    that is saved is the numerical value, not the signs ('++' etc.) that are 
    only meant for the user to understand/remember the meaning.
    In this example:
        * score of 3 means the strongest Like
        * -3 is the strongest Pass
        * 1 is the weakest Like, but still triggers a Like in the site...

Results structure - profiles_df pandas DataFrame:
    * columns: ['profile id',score_column_name,'name','age','location','OKCupid match %',
                'essay dict','basic details','extended details',
                'image filenames','timestamp']
    * 'profile id', a unique OK Cupid identifier that links to the 
        profile URL: https://www.okcupid.com/profile/profile_id, for example 
        profile_id=2636426633407507712 or profile_id=yafchuk.
    * essay_dict is a dictionary of all the free text written in the 
        "essay section", with keys according to the site titles, 
        for example 'My self-summary',"What I'm doing with my life",
        'The first thing people notice about me', etc. Text is kept in 
        html format and should be parsed by sentences, paragraphs, 
        tokenized to language(s) and Emojis...
            

This script can be run entirely, or executed by sections 
    (for example in Spyder).

Requirements:
    * regular Python packages (see imports below)
    * bs4, selenium!
    * chrome browser installed on the computer, and the chromedriver.ext that 
        matches its version (download from http://chromedriver.chromium.org/ and 
        place in a local folder, write its path in chromedriver_path parameter
        below!)
"""

#%% parameters

data_folder_path='D:\AI Data\DatingAI\Data'
images_folder_path='D:\AI Data\DatingAI\Data\Images'
chromedriver_path=r'C:\Program Files (x86)\ChromeDriver\chromedriver.exe'
score_levels=5
account_events={'date format':'%d/%m/%Y',
                'name-date tuples':[
                        ('re-opened free account','19/07/2019'),
                        ('started subscription','11/09/2019'),
                        ('finished subscription','11/10/2019'),
                        ('re-opened free account','22/11/2019'),
                        ('re-opened free account','02/12/2019'),
                        ('re-opened free account','09/12/2019'),
                        ]}
user_no_go_basic_detils=['Married','Divorced','Has kids']
user_no_go_extended_detils=[', Smokes cigarettes',', Smokes marijuana', # the comma to avoid recgonizing 'smokes marijuana' in 'never smokes marijuana'
                            ', Does drugs']

#%% imports
import logging
logging.basicConfig(format='%(asctime)s %(funcName)s (%(levelname)s): %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger('data logger')
logger.setLevel(logging.INFO)

import os
import filecmp
import re
import pandas as pd
import matplotlib.pyplot as plt
import math
import collections

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from bs4 import BeautifulSoup
import urllib.request

#%% definitions
def create_selenium_chromedriver(chromedriver_path,mode='normal',
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        max_timeout_sec=0):
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-logging')
    options.add_argument('--log-level=3')
    options.add_argument("start-maximized")
    options.add_argument("user-agent=%s") # to find a normal user agent - Google 'whats my user agent' on a computer...
    if mode=='minimal':
        options.add_argument("disable-infobars")
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
    
        prefs = {"profile.managed_default_content_settings.images": 2, 'permissions.default.stylesheet': 2}
        options.add_experimental_option("prefs", prefs)
    
    driver=webdriver.Chrome(executable_path=chromedriver_path,options=options)
    if max_timeout_sec>0:
        driver.set_page_load_timeout(max_timeout_sec)
    return driver

def find_all(list_,target):
    """returns starting_indices_list, a list of starting indices of all 
        occurances of the target in list_. if there is no occurance, returns []
        
    target can be a single element (i.e. a character) or a list (i.e. a string)
    """
    indices_list=[]
    for i in range(len(list_)-len(target)+1):
        current=list_[i:i+len(target)]
        if current==target:
            indices_list.append(i)
    return indices_list

def printtify(soup_obj): print(soup_obj.prettify())

def extended_profile_review(profile_index,df,images_folder_path,
                            images_per_row=3):
    """created on 2019/02/08 to present an extended profile review of
        basic profile features AND all images!
        
    df is the dataframe for profiles, assumed to contain columns:
        'profile id','score (levels=%d)'%score_levels,'name','age',
        'location','image filenames'
    """
    assert images_per_row>0, 'images_per_row must be strictly positive'
    plt.figure()
    profile_series=df.iloc[profile_index,:]
    profile_id=profile_series['profile id']
    profile_name=profile_series['name']
    # finding score column (column name in the format of 'score (levels=%d)'%score_levels)
    score_column_name=None
    for column in profiles_df.columns:
        if 'score' in column:
            score_column_name=column
            break
    if score_column_name==None:
        raise RuntimeError("no existing column name in profiles_df contains 'score'!")
    profile_score=profile_series[score_column_name]
    
    profile_age=profile_series['age']
    profile_location=profile_series['location']
    image_filenames=profile_series['image filenames']
    images_num=len(image_filenames)
    columns_num=math.ceil(images_num/images_per_row)
    for idx,image_filename in enumerate(image_filenames):
        plt.subplot(columns_num,images_per_row,idx+1)
        image_array=plt.imread(os.path.join(images_folder_path,image_filename))
        plt.imshow(image_array)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    plt.suptitle('review for profile id: %s\nname: %s, score: %d, age: %d, location: %s'%(
            profile_id,profile_name,profile_score,profile_age,profile_location))
    plt.show()

def discrete_hist(x):
    """plots the discrete histogram of x values - counts of each unique value 
        of x, with no binning! (unlike a regular histogram).
        plotting with double y axis - counts, and % - counts normalized 
        by total count, returning ax1,ax2 axis objects for both y axes
    """
    counter=collections.Counter(x)
    values=list(counter)
    counts_list=[counter[score] for score in values]
    total_counts=sum(counts_list)
    counts_list_normed_percents=[100*score/total_counts for score in counts_list]
    fig, ax1 = plt.subplots()
    ax1.bar(values,counts_list)
    ax1.set_ylabel('counts')
    ax2 = ax1.twinx()
    ax2.bar(values,counts_list_normed_percents)
    ax2.set_ylabel('portion (%)')
    plt.xlabel('unique values')
    ax1.grid(which='major',axis='y')
    return ax1,ax2

#%% opening OK Cupid DoubleTake in a chrome browser by selenium
if not os.path.exists(images_folder_path):
    create_folder_decision=input('images folder does not exist in supplied path (%s), create it there automatically now or abort? [y]/n '%images_folder_path)
    if create_folder_decision!='n':
        os.makedirs(images_folder_path)
    else:
        raise RuntimeError('user chose to abort')
if not os.path.exists(data_folder_path):
    create_folder_decision=input('images folder does not exist in supplied path (%s), create it there automatically now or abort? [y]/n '%data_folder_path)
    if create_folder_decision!='n':
        os.makedirs(data_folder_path)
    else:
        raise RuntimeError('user chose to abort')

# creating score dictionary
if score_levels>0:
    score_dict={-score_levels+idx:'-'*(score_levels-idx) for idx in range(0,score_levels)}
    score_dict.update({idx:'+'*idx for idx in range(1,score_levels+1)})
else:
    raise RuntimeError('non-legit score_levels input, must be >0')

# opening chromedriver with selenium
if 'driver' in locals():
    selenium_decision=input('selenium driver already exists, are you already logged in to OK Cupid and do you want to use it? [y]/n ')
    if selenium_decision!='n':
        driver.get('https://www.okcupid.com/doubletake') # this is no error, if 'driver' in locals() this must work fine...
if ('driver' not in locals()) or ('driver' in locals() and selenium_decision=='n'):
    logger.info('opening a selenium chrome driver and navigating to OK Cupid login page, then login to your account and return here for further instructions')
    driver=create_selenium_chromedriver(chromedriver_path)
    driver.get('https://www.okcupid.com/login')
    if input('only if you have already logged in, hit enter to continue')=='':
        logger.info('navigating to DoubleTake to start scoring and scraping!')
    driver.get('https://www.okcupid.com/doubletake')

#%% scraping OK Cupid DoubleTake sequentially (looping)
profiles_df_path=os.path.join(data_folder_path,'profiles_df.pickle')
score_column_name='score (levels=%d)'%score_levels

while 1:
    # static scraping
    logger.info('starting static scraping')
    soup=BeautifulSoup(driver.page_source,'lxml')
    timestamp=pd.Timestamp.now()
    profile_id=score=name=age=location=match_percents=essay_dict=\
        basic_profile_details=extended_profile_details=None
    skipping_decision='n'
    identical_skipping=False
    # getting soup and profile id
    try:
        soup_card=soup.find('div',class_='cardsummary')
        profile_URL=soup_card.find('a')['href'] # something like https://www.okcupid.com/profile/5605047242975505858?cf=quickmatch, or https://www.okcupid.com/profile/Dor_La?cf=quickmatch
        profile_id=re.search(r'(?<=profile/).*?(?=\?)',profile_URL).group()

        if os.path.isfile(profiles_df_path): # the profiles_df already exists
            profiles_df=pd.read_pickle(profiles_df_path)
            if profile_id in profiles_df.index:
                skipping_decision=input('this profile id was already scraped, do you want to skip (and Like/Pass by score already given) or scrape again and append to data? [y]/n ')
                if skipping_decision!='n': skipping_decision='y' # since the default for anything other than 'n' is 'y'
    except:
         logger.error('fatal error in scraping the card, re-execute to continue. exception error:')
         raise
    
    if skipping_decision=='y':
        score=profiles_df.loc[profile_id][score_column_name] # to be used in Like/Pass
    else:
        # scraping header card details (name,age,location,...)
        try:
            name=soup_card.find('div',class_='cardsummary-item cardsummary-realname').text
        except:
            logger.warning('could not scrape name')
        try:
            age=int(soup_card.find('div',class_='cardsummary-item cardsummary-age').text)
        except:
            logger.warning('could not scrape age')
        try:
            location=soup_card.find('div',class_='cardsummary-item cardsummary-location').text
        except:
            logger.info('could not scrape location')
        try:
            match_text=soup_card.find('span',class_='cardsummary-match-pct').text
            match_percents=int(match_text[:match_text.find('%')])
        except:
            logger.debug('could not scrape match percents')
    
        try: # scraping profile details (orientation, languages,...)
            soup_profile_details=soup.find('div',class_='quickmatch-profiledetails matchprofile-details')
            soup_profile_details=soup_profile_details.find_all('div',class_='matchprofile-details-text')
            basic_profile_details=soup_profile_details[0].text
            if len(soup_profile_details)>1: extended_profile_details=soup_profile_details[1].text
        except:
            logger.debug('could not scrape profile details')
        try: # scraping all free text boxes ("essay")
            soup_essay=soup.find('div',class_='qmessays')
            soup_free_text_boxes=soup_essay.find_all('div',class_='qmessays-essay')
            essay_dict={}
            for soup_box in soup_free_text_boxes:
                box_title=soup_box.find('h2').text
                box_text=str(soup_box.find('p'))
                essay_dict.update({box_title:box_text})
        except:
            logger.debug('could not scrape self summary')
        try: # scraping images!
            soup_carousel=soup.find('div',class_='qmcard-carousel-viewport-inner')
            soup_images=soup_carousel.find_all('img')
            image_urls=[soup_image['src'] for soup_image in soup_images]
            image_filenames=[]
            identical_images_number=0
            for image_url in image_urls:
                slash_indices=find_all(image_url,'/')
                image_url_from_last_slash=image_url[slash_indices[-1]+1:]
                image_name=image_url_from_last_slash[:image_url_from_last_slash.find('.')]
                image_extension=image_url_from_last_slash[image_url_from_last_slash.find('.')+1:image_url_from_last_slash.find('?')]
                image_filename=image_name+'.'+image_extension
                if os.path.isfile(os.path.join(images_folder_path,image_filename)):
                    urllib.request.urlretrieve(image_url,os.path.join(images_folder_path,'temp_image_file'))
                    file_path_a=os.path.join(images_folder_path,image_filename)
                    file_path_b=os.path.join(images_folder_path,'temp_image_file')
                    if filecmp.cmp(file_path_a,file_path_b,shallow=False):
                        logger.info('%s image file already exists and identical in content to the downloaded image file with the same name'%image_filename)
                        identical_images_number+=1
                    else:
                        logger.warning('%s image file already exists but not identical in content to the downloaded image file  with the same name, renaming to %s'%(
                                image_filename,image_name+'_2'))
                        image_name=image_name+'_2'
                        image_filename=image_name+'.'+image_extension
                        urllib.request.urlretrieve(image_url,os.path.join(images_folder_path,image_filename))
                else:
                    urllib.request.urlretrieve(image_url,os.path.join(images_folder_path,image_filename))
                image_filenames.append(image_filename)
        except:
            logger.warning('could not scrape images')
        logger.info('profile successfully scraped')
        
        if identical_images_number==len(image_urls):
            for idx,profile in profiles_df.iterrows():
                if (profile['age']==age and profile['location']==location and \
                    profile['essay dict']==essay_dict and \
                    profile['basic details']==basic_profile_details and \
                    profile['extended details']==extended_profile_details and \
                    profile['image filenames']==image_filenames):
                        logger.warning('found a profile with identical details as those scraped now -> skipping profile (Like/Pass by the score already given)!')
                        score=profile[score_column_name]
                        identical_skipping=True
                        break
                
        if identical_skipping==False:
            
            # checking user no-go's
            if isinstance(basic_profile_details,str):
                for no_go in user_no_go_basic_detils:
                    if no_go in basic_profile_details:
                        logger.warning("detected user basic details no-go in profile: '%s'"%(no_go))
            
            if isinstance(extended_profile_details,str):
                for no_go in user_no_go_basic_detils:
                    if no_go in basic_profile_details:
                        logger.warning("detected user extended details no-go in profile: '%s'"%(no_go))
            
            # acquiring score from the user
            nonlegit_score=True
            while nonlegit_score:
                if name==None:
                    text_for_completion='enter your score for profile by %s or enter 0 to break: '%(score_dict)
                else:
                    text_for_completion='enter your score for %s by %s or enter 0 to break: '%(name,score_dict)
                try:
                    score=float(input(text_for_completion))
                except:
                    logger.error('conversion of input to float failed!')
                
                if score!=None and ((score in score_dict) or score==0):
                    nonlegit_score=False
                else:
                    logger.error('non-legit score given, repeating')
                    
            
            if score==0:
                for image_filename in image_filenames:
                    os.remove(os.path.join(images_folder_path,image_filename))
                logger.info('user gave score=0 -> deleted last downloaded images, breaking')
                break
            # saving results
            if ('profiles_df' in locals()) or os.path.isfile(profiles_df_path):
                if 'profiles_df' not in locals(): profiles_df=pd.read_pickle(profiles_df_path)
                # checking if the existing profiles_df was built with the same score levels as current score_levels
                existing_columns=profiles_df.columns
                existing_score_column_name=None
                for column in existing_columns:
                    if 'score' in column:
                        existing_score_column_name=column
                        break
                if existing_score_column_name==None:
                    raise RuntimeError("no existing column in profiles_df contains 'score', data is corrupted!")
                try:
                    existing_score_levels=int(re.search(r'[0-9]',existing_score_column_name).group())
                except:
                    raise RuntimeError("%s exists, but could not extract its score levels from the first column (should be named 'score (levels=%%d))'%%score_levels), rename or delete the file and re-execute to build a new profiles_df"%profiles_df_path)
                if existing_score_levels!=score_levels:
                    raise RuntimeError('%s exists with score_levels=%d, current input is score_levels=%d -> adjust score levels OR manually rename or delete the existing file -> re-execute to build a new profiles_df'%(
                            profiles_df_path,existing_score_levels,score_levels))
                else:
                    current_df=pd.DataFrame.from_dict({len(profiles_df):{
                            'profile id':profile_id,
                            score_column_name:score,
                            'name':name,
                            'age':age,
                            'location':location,
                            'OKCupid match %':match_percents,
                            'essay dict':essay_dict,
                            'basic details':basic_profile_details,
                            'extended details':extended_profile_details,
                            'image filenames':image_filenames,
                            'timestamp':timestamp}},
                        orient='index')
                    profiles_df=pd.concat([profiles_df,current_df])
            else: # no profiles_df already exists
               building_decision=input('%s does not exist, build a new file? [y]/n '%(profiles_df_path))
               if building_decision=='n':
                   raise RuntimeError('aborted according to user decision')
               else:
                   profiles_df=pd.DataFrame.from_dict({0:{
                            'profile id':profile_id,
                            score_column_name:score,
                            'name':name,
                            'age':age,
                            'location':location,
                            'OKCupid match %':match_percents,
                            'essay dict':essay_dict,
                            'basic details':basic_profile_details,
                            'extended details':extended_profile_details,
                            'image filenames':image_filenames,
                            'timestamp':timestamp}},
                        orient='index')
            # saving profiles_df
            profiles_df.to_pickle(profiles_df_path)
            logger.info('%s successfully updated'%profiles_df_path)
    
    # Like/Pass the profile according to given user score to pass decision to OK Cupid, and continue scraping
    html=driver.find_element_by_tag_name('html')
    if score>0:
        msg='score <%d> was given -> Liked (NUMPAD2 hit). %d profiles scraped!'%(score,len(profiles_df))
#        driver.find_element_by_xpath('//*[@id="quickmatch-wrapper"]/div/div/span/div/div[2]/div/div[2]/span/div/div/div/div[1]/div[2]/button[2]').click()
        html.send_keys(Keys.NUMPAD2)
    else:
        msg='score <%d> was given -> Passed (NUMPAD1 hit). %d profiles scraped!'%(score,len(profiles_df))  
#        driver.find_element_by_xpath('//*[@id="quickmatch-wrapper"]/div/div/span/div/div[2]/div/div[2]/span/div/div/div/div[1]/div[2]/button[1]').click()
        html.send_keys(Keys.NUMPAD1)
    
    logger.info(msg)

# exporting to excel for easy user review
profiles_excel_path=os.path.join(data_folder_path,'profiles_df.xlsx')
profiles_df.to_excel(profiles_excel_path)
logger.info("exported all scraped profiles data for easy user review to '%s'"%(profiles_excel_path))

#%% short review of results
total_num_of_images=sum([len(image_filenames) for image_filenames in profiles_df['image filenames']])
logger.info('scraped %d profiles so far, containing %d images in total, plotting score histogram'%(
        len(profiles_df),total_num_of_images))

# plotting score distribution
#profiles_df[score_column_name].hist(rwidth=0.9) # only use if scores are non-integer, otherwise use:
ax1,ax2=discrete_hist(profiles_df[score_column_name])
ax1.set_ylabel('times score given')
ax2.set_ylabel('times score given (%)')
plt.xlabel('score')
plt.title('discrete distribution of scores given\ntotal profiles scraped and scored: %d'%len(profiles_df))

# checkup: extended_profile_review for the last profile
logger.info('checkup: plotting the profile image of the last profile scraped with some basic data')
extended_profile_review(profile_index=len(profiles_df)-1,
             df=profiles_df,images_folder_path=images_folder_path,
             images_per_row=3) 

#%% analyzing rolling score
rolling_window=round(0.03*len(profiles_df)) # int, 0=no rolling, else the number of observations to merge
#min_date=pd.to_datetime('25/05/2019',format="%d/%m/%Y")
# end of inputs ---------------------------------------------------------------

plt.figure()
score_series=profiles_df[score_column_name]
score_series.index=profiles_df['timestamp']
score_rolling_mean=score_series.rolling(rolling_window).mean()
if rolling_window<=0:
    score_series.plot()
else:
    plt.plot(score_rolling_mean.index,score_rolling_mean,'-x',
             linewidth=2,markersize=3,
             label='score rolling mean (period: %d points)'%(rolling_window))

for name,date in account_events['name-date tuples']:
    plt.plot(pd.to_datetime([date,date],format=account_events['date format']),
             [-score_levels,score_levels],'--',label=name)

#plt.xlim(left=min_date)
plt.legend(loc='best')
plt.ylabel('score')
plt.title('user personal taste score on profiles presented by OKCupid')
plt.grid()

fig_path=os.path.join(data_folder_path,'personal_taste_fig.png')
plt.savefig(fig_path)
logger.info("saved personal taste figure to '%s'"%(fig_path))

#%% verifying user no-go's in profiles_df
num_of_detected_no_gos_profiles=0
for idx,profile in profiles_df.iterrows():
    no_go_profile=False
    if isinstance(profile['basic details'],str):
        for no_go in user_no_go_basic_detils:
            if no_go in profile['basic details'] and profile[score_column_name]>0:
                logger.info("profile %d got score=%d but its basic details contain user no-go ('%s'):\n\t%s"%(
                        idx,profile[score_column_name],no_go,profile['basic details']))
                no_go_profile=True
    
    if isinstance(profile['extended details'],str):      
        for no_go in user_no_go_extended_detils:
            if no_go in profile['extended details'] and profile[score_column_name]>0:
                logger.info("profile %d got score=%d but its extended details contain user no-go ('%s'):\n\t%s"%(
                        idx,profile[score_column_name],no_go,profile['extended details']))
                no_go_profile=True
    if no_go_profile:
        num_of_detected_no_gos_profiles+=1
logger.info("\n\ncompleted user no-go's verification, detected profiles to contain no-go's: %d (%.1f%% of total)"%(
        num_of_detected_no_gos_profiles,100*num_of_detected_no_gos_profiles/len(profiles_df)))