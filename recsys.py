# Imports
import spotipy
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

# Acessing Spotipy
config = configparser.ConfigParser()
config.read('config.cfg')
client_id = config.get('SPOTIFY', 'CLIENT_ID')
client_secret = config.get('SPOTIFY', 'CLIENT_SECRET')
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))

# Test tracks, Min 5 input songs for PCA to work properly!
input_tracks = ['https://open.spotify.com/track/2IxhiriDpu4iBnXZb3ytXN?si=3e118fa0c7564ae9', 
'https://open.spotify.com/track/3m7CG7IaZffxYKSPljcw7E?si=42d1fff356f1419c', 
'https://open.spotify.com/track/2YByIMqNtTb0T072UDfTo9?si=676f09958ce749a5', 
'https://open.spotify.com/track/2fMCh2xOtwMt1S8iV2Laok?si=30fac17559f3479e', 
'https://open.spotify.com/track/3lUQ6y8XeeaoK2hPydcX9c?si=c7c16a25b3074d25'] 

print(input_tracks[0])

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

# Mud racing:
cos_sim_list = np.empty((num_input_s, len(track_features_data)))


for i in range(num_input_s):
    a = audio_features_np[i].reshape(1, -1)
    cos_sim = cosine_similarity(a,track_features_data)
    cos_sim_list[i] = cos_sim[0]

summed_cos_sim = np.sum(cos_sim_list, axis= 0)

rec_songs = []
num_rec_songs = 5


for i in range(num_rec_songs):
    ind_max = np.where(summed_cos_sim == max(summed_cos_sim))
    id_ind_max = track_features_data_with_id.loc[ind_max[0], ['id']].iloc[0]['id']
    rec_songs.append("https://open.spotify.com/track/" + id_ind_max)
    summed_cos_sim[ind_max] = 0

# Weight with spotifys own recommendations
sp_rec_songs = spotify.recommendations(seed_tracks = [input_tracks[0]])
audio_features_sp_rec_songs = spotify.audio_features(sp_rec_songs['tracks'][1]['uri'])
n_rec_songs_sp = len(sp_rec_songs['tracks'])
audio_features_np_sp_rec_songs = np.empty((n_rec_songs_sp,11))

print(spotify.audio_features(sp_rec_songs['tracks'][1]['uri']))

for i in range(len(audio_features_np_sp_rec_songs)):
    for j in range(len(relevant_features)):
        audio_features_sp = spotify.audio_features(sp_rec_songs['tracks'][i]['uri'])
        print(audio_features_sp)
        audio_features_np_sp_rec_songs[i][j] = audio_features_sp[0][relevant_features[j]]
        j = j + 1
    i = i + 1

pca1 = PCA(n_components = n_comp)
audio_features_np_sp_rec_songs = pca1.fit_transform(audio_features_np_sp_rec_songs)

# Calculate the cosine sim between our recommended songs and spotifys recommended songs. Songs we recommended that get a high cosine similarity get weighted higher.
cos_sim_list_sp = np.empty((len(audio_features_np_sp_rec_songs), 1))

a = audio_features_np.reshape(1, -1)
cos_sim_sp = cosine_similarity(a,track_features_data)
cos_sim_list_sp = cos_sim_sp

# https://open.spotify.com/track/5b9k3fEryRGfcdqzJE2DKa
# nollställ input låtar, get id of rec songs.
