# Imports
from numpy.lib.function_base import average
import spotipy
from spotipy.client import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import configparser
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pw
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from numpy import dot, string_
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys

# Acessing Spotipy
config = configparser.ConfigParser()
config.read('config.cfg')
client_id = config.get('SPOTIFY', 'CLIENT_ID')
client_secret = config.get('SPOTIFY', 'CLIENT_SECRET')
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))

# Test tracks, Min 5 input songs for PCA to work properly!, alt rock chill time
input_tracks = ['https://open.spotify.com/track/2N4idqj9TT3HnH2OFT9j0v?si=466cc7f18ac84d16', 
'https://open.spotify.com/track/4Iyo50UoYhuuYORMLrGDci?si=dab4bc69a1214552', 
'https://open.spotify.com/track/1bAZV1EBTRi9t1cVg75i8t?si=6bfb8d07c10b4a16', 
'https://open.spotify.com/track/0J0UZpA2Ivp4qaXe3QzCrT?si=220416549bac43ec', 
'https://open.spotify.com/track/3Nk3CL1Z73VMydXFCfnTcI?si=2dcc74f6124a4f5d']

num_input_s = len(input_tracks)

# Inserting relevant features into numpy array
audio_features = spotify.audio_features(input_tracks)
input_track_ids = np.empty((num_input_s, 1), dtype='object')
audio_features = np.array(audio_features)
audio_features_np = np.empty((len(audio_features),11))
i = 0
relevant_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for i in range(len(audio_features)):
    for j in range(len(relevant_features)):
        audio_features_np[i][j] = audio_features[i][relevant_features[j]]
        input_track_ids[i] = audio_features[i]['id']
        j = j + 1
    i = i + 1

# Test data set
track_features_data = pd.read_csv('D:\Skola\ML\\tracks_features.csv')
track_features_data_with_id= track_features_data.loc[:, ['id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' ]]
track_features_data= track_features_data.loc[:, ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' ]]

# Scaling the data
sc1 = StandardScaler().fit(audio_features_np)
sc2 = StandardScaler().fit(track_features_data)
audio_features_np = sc1.transform(audio_features_np)
track_features_data = sc2.transform(track_features_data)

# PCA
n_comp = min(num_input_s, len(relevant_features))
from sklearn.decomposition import PCA
pca1 = PCA(n_components = n_comp)
track_features_data = pca1.fit_transform(track_features_data)

pca2 = PCA(n_components = n_comp)
audio_features_np = pca2.fit_transform(audio_features_np)

cos_sim_list = np.empty((num_input_s, len(track_features_data)))
print(np.shape(audio_features_np))

for i in range(num_input_s):
    for j in range(len(track_features_data)):
        a = audio_features_np[i].reshape(1, -1)
        cos_sim_list[i][j] = distance.cityblock(a,track_features_data[j])

summed_cos_sim = np.sum(cos_sim_list, axis= 0)

print(summed_cos_sim, np.shape(summed_cos_sim))

rec_songs = []
num_rec_songs = 1000
rec_songs_features = np.empty((num_rec_songs,n_comp + 3)) # 5 Principal components : Index : Summed cos sim : Main genre 
rec_songs_index_cos = np.empty((num_rec_songs,2))

for i in range(num_rec_songs):
    ind_max = np.where(summed_cos_sim == min(summed_cos_sim))
    rec_songs_features[i][5] = ind_max[0][0]
    rec_songs_features[i][6] = max(summed_cos_sim)
    rec_songs_features[i][0:5] = track_features_data[ind_max][0]
    id_ind_max = track_features_data_with_id.loc[ind_max[0], ['id']].iloc[0]['id']
    rec_songs.append("https://open.spotify.com/track/" + id_ind_max)
    summed_cos_sim[ind_max] = 1000

rec_songs_genres = []
rec_songs_id_score = np.empty((num_rec_songs, 2), dtype = object) # Summed TF-IDF score with input songs genre wise : ID

# Create list with string with genres for each artist
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

for i in range(num_rec_songs):
    rec_songs_id_score[i][1] = rec_songs[i]
    artist_url = spotify.track(rec_songs[i])['album']['artists'][0]['external_urls']['spotify']
    main_genre = ""
    for i in range(len(spotify.artist(artist_url)['genres'])):
        main_genre += spotify.artist(artist_url)['genres'][i] + " "
    rec_songs_genres.append(main_genre)


for i in range(num_input_s):
    artist_url = spotify.track(input_tracks[i])['album']['artists'][0]['external_urls']['spotify']
    main_genre = ""
    for i in range(len(spotify.artist(artist_url)['genres'])):
        main_genre += spotify.artist(artist_url)['genres'][i] + " "
    rec_songs_genres.append(main_genre)

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(rec_songs_genres)


sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


summed_tfidf = np.sum(sim_matrix[:num_rec_songs, num_rec_songs:(num_rec_songs + num_input_s)], axis = 1)
for i in range(num_rec_songs):
    rec_songs_id_score[i][0] = summed_tfidf[i]
print(rec_songs_id_score)

def sort_inner(inner):
    return inner[0]

rec_songs_id_score = sorted(rec_songs_id_score, key = sort_inner, reverse = True)

print(rec_songs_id_score[:5][:])

lim = 20
sp_rec_songs = np.empty(num_input_s * lim, dtype=object)
for i in range(num_input_s):
    s_rec = spotify.recommendations(seed_tracks = [input_tracks[i]], limit = lim)
    for j in range(lim):
        sp_rec_songs[(i * lim) + j] = s_rec['tracks'][j]['external_urls']['spotify']

nr_same_songs = 0
for i in range(num_input_s * lim):
    if (rec_songs_id_score[i][1] in sp_rec_songs):
        nr_same_songs += 1

print("Accuracy compares to spotifys recommendations: ", (nr_same_songs / (num_input_s * lim)))

# 10 min computing time with 1000 songs. All five songs were in line with the input songs.