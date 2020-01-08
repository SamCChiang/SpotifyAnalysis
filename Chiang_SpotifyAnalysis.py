#!/usr/bin/env python
# coding: utf-8

# # Predicting Danceability
# *Sam Chiang for BUS AN 579, December 4, 2019*
# 

# ## Dataset Description

# #### Top Spotify Tracks of 2018
# 
# At the end of each year, Spotify compiles a playlist of the songs streamed most often over the course of that year. This year's playlist (Top Tracks of 2018) includes 100 songs. What features are most popular and how well do they correlate to each other? What features make a song danceable?
# 
# Original Data Source: This dataset was downloaded from Kaggle: https://www.kaggle.com/nadintamer/top-spotify-tracks-of-2018
# The audio features for each song were extracted using the Spotify Web API and the spotipy Python library, making this source very reliable. 
# 
# This dataset includes columns:
# 
# + ID: Spotify URI for the song
# + Name of the song
# + Artist(s) of the song
# + Danceability: describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
# + Energy: a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
# + Key: the key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.
# + Loudness: the overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
# + Mode: indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
# + Speechiness: detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
# + Acousticness: a confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
# + Instrumentalness: predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
# + Liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
# + Valence: a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
# + Tempo: the overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
# + Duration_ms: the duration of the track in milliseconds.

# ## Read-in Dataset

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("/Users/SamChiang/Desktop/MSBA/BUSAN 579/top2018.csv")
df.head() 


# ## Clean Dataset 
# I do not need to know the ID and name column in this analysis because they are not useful in this case.

# In[3]:


drop_list = ["id", "name"] # List of column names

df = df.drop(drop_list, 
        axis=1)

df.info() # Check the column types in the dataframe


# By checking the dataframe info and for missing date, we can see that the columns are non-null and there are no typos in the column names

# In[4]:


# Check again to see if there are any null values in the dataset

missing_data = df.isnull()

num_missing = missing_data.sum()
num_missing


# Another thing I want to change to make things more intuitive is to change the duration of songs from milliseconds to minutes.

# In[5]:


# Divide values in the duration_ms column by 1000 to convert into seconds, then 60 to convert into minutes
df['duration_ms'] = df['duration_ms']/1000/60
df.rename(columns={'duration_ms':'duration'}, inplace=True) # Rename duration_ms header to duration because it is no longer in milliseconds


# ## Binning
# At a glance, duration seems to have the greatest variation so I will try to bin songs together based on how long they are.

# In[6]:


# Check what the shortest song is in minutes
df['duration'].min()


# In[7]:


# Check what the longest song is in minutes
df['duration'].max()


# Based on these min and max findings, I will create a bin for each minute interval starting from 1 to 7 minutes.

# In[8]:


# Create bins using a range function from 1-7 in 1 minute intervals
bins = list(range(1, 8, 1))
bins


# Now I will add labels for the corresponding duration bins and create its new column in the dataframe.

# In[9]:


df["duration_cat"] = pd.cut(df["duration"], bins=bins, labels=["1-2", "2-3", "3-4", "4-5", "5-6", "6-7"])


# In[10]:


df.head()


# How long do artists generally make their songs? By grouping the bins, I can see that 3-4 minute songs are the most popular, with most other songs under 4 minutes.

# In[11]:


# Group by the duration categories and count how many artists have songs belonged in that category
df.groupby("duration_cat")["artists"].count()


# With a histogram, I can confirm this analysis visually where the chart is slightly left skewed. 

# In[12]:


df['duration'].plot(kind='hist', title='Duration Distribution')


# Similarly, I want to bin tempo because it often varies in different songs. Because the bins for tempo is not as intuitive, I will create a histogram to see the distribution of tempo.

# In[13]:


df['tempo'].plot(kind='hist', title='Tempo Distribution')


# In[14]:


# Based off the histogram, I will create bins in intervals of 20 from 60-200
bins_tempo = list(range(60, 220, 20))
bins_tempo


# Now I will add labels for the corresponding tempo bins and create its new column in the dataframe.

# In[15]:


df["tempo_cat"] = pd.cut(df["tempo"], bins=bins_tempo, labels=["60-80", "80-100", "100-120", "120-140", "140-160", "160-180", "180-200"])


# In[16]:


df.head()


# In[17]:


# Group by the tempo categories and count how many artists have songs belonged in that category
df.groupby("tempo_cat")["artists"].count()


# Grouping by tempo categories, I can see that songs vary greatly in terms of tempo. However, most songs fall under 160 bpm, which is consistent with the histogram.

# ## EDA
# Using exploratory data analysis, I want to see correlations and distributions within the data to find what popular music generally consists of. Furthermore, I want to use these findings to show what makes a song danceable.

# How many artists are in each duration category? Thus, what song durations are the most popular for artists?

# In[18]:


df.groupby("duration_cat")["artists"].count()


# #### Univariate Analysis
# Provide summary information about the distribution of data in each column

# In[19]:


df.describe()


# The average danceability rating for these spotify songs is about .72 with a standard deviation of .13, which means most songs are relatively danceable. 

# #### Visualize Distributions
# 
# 

# I can visualize some of these distributions with the following:

# In[20]:


df[["danceability"]].boxplot()


# With this boxplot, I can see that there are a few outliers where some sounds are below the 0.3 rating. However, because I am more focused on what makes a song danceable and because every song is unique, I will not omit these points. My next questions would involve what attributes contibute to a danceable song.

# **Correlation between variables**
# 
# I want to see correlations between certain atrributes to danceability and tempo. In my opinion, the most intuitive attributes that would influence danceability are tempo, valence, energy, loudness, and speechiness.

# In[21]:


# Duration and danceability are negatively correlated, so the longer a song is the less likely it will be danceable
df[['danceability', 'duration']].corr()


# In[22]:


# Tempo and danceability are negatively correlated, so the slower the song is the more danceable 
df[['danceability', 'tempo']].corr()


# In[23]:


# Valence and danceability are positively correlated, so the higher valence, the higher danceability
df[['danceability', 'valence']].corr()


# In[24]:


# Energy and danceability are negatively correlated, so the less energetic, the more danceabled
df[['danceability', 'energy']].corr()


# In[25]:


# Loudness and danceability are slightly positively correlated, so the louder a song is, the more likely it is danceable
df[['danceability', 'loudness']].corr()


# In[26]:


# Speechiness and danceability are positvely correlated, so the more wordy a song, the more likely it will be danceable
df[['danceability', 'speechiness']].corr()


# Of the few correlation analyses, speechiness, valence, and tempo have the strongest correlations to danceability, which are all positive. Further, I can view all the correlations as a whole with a visualization. Upon observation, one of the most positively correlated attributes is loudness and energy, which makes sense. On the other side, the least correlated is energy and acousticness, which might make sense because acoustic songs generally have a steadier pace.

# In[32]:


import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
    xticklabels=corr.columns.values, # set x labels as the column names
    yticklabels=corr.columns.values, # set y labels as the column names
    vmin=-1, # set the minimum vertical value to -1
    vmax=1, # set the max vertical value to 1
    center=0, # set the middle of the scale at the value of 0
    cmap=sns.diverging_palette(10, 300, n=200), # set the color scheme where there it diverges into two colors
    square=True) # set the shape of each attribute to a square shape


# Because the scales of these attributes vary in types and range, I will normalize them and compare the distributions once again.

# In[28]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(df[["speechiness", "tempo", "valence"]],
    df[["danceability"]], # input x=features, y=targets
    test_size = 0.3) # set aside 30% for testing

pd.DataFrame(xTrain).hist()  # convert feature array to dataframe and plot histogram


# In[29]:


# Create the Scaler object
from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()

# apply the transformation to the training data
X_train_std = std_scaler.fit_transform(xTrain) # apply the transformation to the testing data
X_test_std = std_scaler.transform(xTest) # but we only transform our testing data with already fit scaler

# convert resulting array back to dataframe
Xtest_std = pd.DataFrame(X_test_std,
                     columns=xTrain.columns)

Xtest_std.head()


# With a much more comparable scale, it allows for easier comparison across features. Let's check their distributions: 

# In[30]:


Xtest_std.describe()


# In[33]:


import seaborn as sns
corr_std = Xtest_std.corr()
sns.heatmap(corr_std, 
    xticklabels=corr_std.columns.values, # set x labels as the column names
    yticklabels=corr_std.columns.values, # set y labels as the column names
    vmin=-1, # set the minimum vertical value to -1
    vmax=1, # set the max vertical value to 1
    center=0, # set the middle of the scale at the value of 0
    cmap=sns.diverging_palette(10, 300, n=200), # set the color scheme where there it diverges into two colors
    square=True) # set the shape of each attribute to a square shape


# With this narrowed down heat map, I can see a clearer visual correlation matrix. 

# ### Model Implementation

# I want to be able to use multiple linear regression to predict whether a song is danceable or not. My predictors include speechiness, tempo, and valence.

# In[64]:


from sklearn import linear_model

# instantiate the model object
lm = linear_model.LinearRegression()

# fit the model to the dataset
lm_model = lm.fit(xTrain, yTrain)

# Make predictions using the testing set
y_pred = lm.predict(xTest)
print(y_pred) # See the predictions


# In[48]:


# Create data frame with prediction results
pred_df = pd.DataFrame(yTest) 
pred_df["preds"] = linear_Pred

pred_df.head()


# In[74]:


import matplotlib.pyplot as plt

df1 = pred_df.head(25) # See the first 25 rows
df1.plot(kind='bar',figsize=(10,8)) # Plot a bar graph for the data set
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# What this visualization tells about our predictions:

# In[83]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(pred_df["danceability"], pred_df["preds"]))
print('Mean Squared Error:', metrics.mean_squared_error(pred_df["danceability"], pred_df["preds"]))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(pred_df["danceability"], pred_df["preds"])))


# The root mean squared error (RMSE), or the standard deviation of the unexplained variance, is 0.11, which is quite low. This means that the model is relatively a good fit.

# ### Conclusion

# I started off this project wanting to understand the general trends in the most popular songs on Spotify. 
# 
# I found that the shortest song was about 1.59 minutes while the longest song is 6.96 minutes. Most songs are between 3 to 4 minutes and are rarely over 5 minutes long. Song tempos, on the other hand, are widely distributed. The most common tempo is between 80-160 bpm.
# 
# This brings me to wonder about danceability, which becomes the target variable, described as how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. Danceability is usually rated between 0.65-0.8. There are some outliers with songs under 0.35, but they aren’t omitted.
# 
# Next, I wanted to see what attributes correlated to danceability. Duration was slightly negatively correlated. Tempo was very negatively correlated. Valence was very positively correlated. Energy was surprisingly slightly negatively correlated. Loudness was a only lightly positively correlated, which was also a bit suprising. Speechiness was positively correlated.
# 
# I chose the most correlated attributes to create a multiple linear regression to predict a danceable song using sklearn.
# 
# By plotting a comparison between the danceability ratings and predictive ratings, I can tell that they match fairly well with small gaps between them.
# 
# Along with an understanding of the errors, especially the root mean squared error of 0.11, it tells me the model predicts with a high level of accuracy.
# 
# With this new understanding, some questions for next time would be to do a deeper analysis in finding why energy negatively correlates to danceability because that sounds intuitively contradicting. 
