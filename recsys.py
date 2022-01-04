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

# Test tracks
input_tracks = ['https://open.spotify.com/track/5hmek3mrSYvfSElBsPNbxo?si=a1c8f7d350744f77', # Immortal Rites, Morbid Angel, Death Metal
'https://open.spotify.com/track/4dzBqc5t2GWJKpGqUoTbrU?si=79152cc8d9a246aa', # 2gether, Muramasa, Electronic
'https://open.spotify.com/track/1bAZV1EBTRi9t1cVg75i8t?si=64a754918b6142de', # I want wind to blow, The Microphones, Indie folk
'https://open.spotify.com/track/1k1Bqnv2R0uJXQN4u6LKYt?si=16ed38939aa645c5', # Ain't no sunshine, Bill Withers, Acoustic soul
'https://open.spotify.com/track/3e9HZxeyfWwjeyPAMmWSSQ?si=66c86570531e44af'] # Thank you, next, Ariana Grande, Pop

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

# Mud racing:
cos_sim_list = np.empty((num_input_s, len(track_features_data)))
print(np.shape(cos_sim_list))


for i in range(num_input_s):
    a = audio_features_np[i].reshape(1, -1)
    b = track_features_data
    cos_sim = cosine_similarity(a,b)
    print(np.shape(cos_sim))
    cos_sim_list[i] = cos_sim[0]

summed_cos_sim = np.sum(cos_sim_list, axis= 0)

rec_songs = np.empty((1, 5))
print(np.shape(rec_songs))

# nollställ input låtar, get id of rec songs.

# for i in range(len(rec_songs)):
#     # rec_songs[i] = 

# OLD:

# # Cluster songs depending on euclidian distance in feature space
# dist = 0
# num_clusters = len(audio_features)
# while dist < 50 and num_clusters > 2:
#     temp_dist = 1e7
#     kmeans = KMeans(n_clusters=num_clusters, random_state= 0).fit(audio_features_np)
#     for k in range(num_clusters):
#         for l in range(num_clusters):
#             if distance.euclidean(kmeans.cluster_centers_[k], kmeans.cluster_centers_[l]) < temp_dist:
#                 temp_dist = distance.euclidean(kmeans.cluster_centers_[k], kmeans.cluster_centers_[l])
#     dist = temp_dist
#     num_clusters = num_clusters - 1
