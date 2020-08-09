# DatingAI
**(last update: 2020/05/31)**

* *Around the beginning of 2019* I started **DatingAI** as a personal research project: learning personal taste in dating sites (by deep learning, mainly computer vision & NLP), developing a personal dating agent.

* *Around the end of 2019* I also started **DatingAI community project**: developing an AI system to assist in *finding relationships*, together with the community I’ve built, which is the opposite of online dating apps that make a fortune from *keeping users single* to watch ads or pay subscriptions as long as possible. You can read more [here](https://ozzisphere.wordpress.com/2019/09/29/datingai-the-community-matchmaking-phase/). We are working on it in [DatingAI Facebook group (in Hebrew)](https://www.facebook.com/groups/DatingAI/).

**This repo only belongs to my DatingAI personal research project, not the community project!**

## About me
I am originally a physicist – B.Sc. from the Technion, M.Sc. from the Weizmann Institute. In 2018 I decided to stop a Ph.D. in astrophysics in order to pursue my new passion – researching AI in the hi-tech! **I have broad research and development experience in data science and machine learning (classic & deep), especially in the fields of NLP, computer vision & time series.**

I am an excellent autodidact, very curious, **I deeply like learning, researching and solving challenges.**

You can read more in www.linkedin.com/in/oz-livneh.

**Feel free to contact me** - in my LinkedIn or oz.livneh@gmail.com.

Yours,

Oz Livneh

<p align="center">
  <img src="Images/DatingAI_logo.jpg">
</p>

# Intro
As written above, **DatingAI** is my personal research project: learning personal taste in dating sites, developing a personal dating agent.

Since OK Cupid is a top global dating site and offers a web interface that sequentially presents profiles to the logged user when liking/passing them (DoubleTake), I built a personal scraper that allows any user to locally save all data that is presented:
<p align="center">
  <img src="Images/Data_example.png">
</p>

**The challenge is to develop and integrate learning:**
1. Profile photos – an arbitrary-length image sequence of a person, possibly with friends, pets...
2. Profile text – an arbitrary-length sequence of weakly-structured optional text sections, in possibly a few intertwined languages, e.g. in Israel – English, Hebrew, Emojis.
3. Categorical features (location, habits…) that require sparse/dense embeddings.
4. Numerical features (age, height…).

# Main neural net architectures

## Learning text with an attentive char-level bi-LSTM ([see notebook](https://github.com/Ozziko/DatingAI/blob/master/Text_attentive_char_biLSTM_score_regression.ipynb))
<p align="center">
  <img src="Images/Text_learning_scheme.png">
</p>

## Learning image sequences with self-attention ([see notebook](https://github.com/Ozziko/DatingAI/blob/master/Image_sequence_self_attentive_score_regression.ipynb))
<p align="center">
  <img src="Images/Image_sequence_learning_scheme.png">
</p>

# Project Roadmap
1. (100%) **Developing a Personal Cupid Scraper**: a Python script that opens a Chrome browser (selenium-driven) for a user to login into the personal OK Cupid account, scrapes all data (textual+images) from the profiles suggested in DoubleTake, then for each profile it asks the user to give a score and accordingly likes/passes the profile and advances to the next profile. All data – text fields, images and the user scores – is saved locally on the disk.

2. (100%) **Developing an image score classification:** for each image, predicting the score class given to the profile to which the image belongs. Developing a PyTorch dataset (with augmentation), train/validation dataloaders, net configurations (mine & pretrained models), training, evaluation metrics, saving/loading weights.

3. (100%) **Developing an image score regression ([see notebook](https://github.com/Ozziko/DatingAI/blob/master/Image_score_regression.ipynb)):** for each image, predicting the score value given to the profile to which the image belongs (with a few net architectures, including Inception v3). *This assumes that all images in each profile are independent and given the same score (of their profile), which is obviously a simplification*. Why regression suits better than classification, as in (2): instead of predicting the image score *class*, this architecture predicts the image score *value* - *to take into consideration the distance between scores in the loss*, e.g. for a target score of +3, predicting -1 (err=4) should be much worse than predicting +2 (err=1), but in classification the penalty on both mistakes is the same.

4. (100%) **Developing a text attentive char bi-LSTM score regression ([see notebook](https://github.com/Ozziko/DatingAI/blob/master/Text_attentive_char_biLSTM_score_regression.ipynb)):** predicting the user score given for each profile based on the free text that is written in the profile, which is weakly-structured - there are many kinds of optional sections to fill freely, e.g. 'My self-summary', "What I'm doing with my life", 'I value', etc. (1) The text on each section is encoded by a character level bi-LSTM, since I deal with text of Israeli profiles - intertwined English, Hebrew & Emojis, where Hebrew is right-to-left, and highly inflected (which makes it highly efficient). (2) An attention net summarizes all sections in each profile, where the inputs for each section are the embedding of the section headline and the bi-LSTM state vector achieved by operating on the section text. (3) A final regression net transforms the final attention output into a profile score.

5. (100%) **Developing an image sequence self-attentive score regression ([see notebook](https://github.com/Ozziko/DatingAI/blob/master/Image_sequence_self_attentive_score_regression.ipynb)):** predicting the user score given for each profile based on the entire *varying-length* sequence of images appearing in the profile. This is a generalization of (3), image score regression, that assumed that the images are independent - predicting *for each image* the score given to the profile to which the image belongs.

6. (10%) **Developing a score regression net for all the rest of the profile features -** all profile features that are not free text or images (e.g. age, location, height, languages, etc.), using embeddings when needed.   

7. (0%) **Integrating (4)-(6) nets into a single regression net**, to be trained together with some attention on final results to find the importance of the different nets, using the different profile features. 

8. (0%) **Researching the representation of personal taste and profile in embedded spaces, clustering, etc.**

8. (0%) **Developing a personal dating agent:** automatically likes/dislikes profiles and messages them according to learned user taste.
