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

# Acessing Spotipy
config = configparser.ConfigParser()
config.read('config.cfg')
client_id = config.get('SPOTIFY', 'CLIENT_ID')
client_secret = config.get('SPOTIFY', 'CLIENT_SECRET')
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))


input_tracks = ['https://open.spotify.com/track/5hmek3mrSYvfSElBsPNbxo?si=23dccac390884bba', # Death metal
'https://open.spotify.com/track/4dzBqc5t2GWJKpGqUoTbrU?si=79152cc8d9a246aa', 
'https://open.spotify.com/track/0J0UZpA2Ivp4qaXe3QzCrT?si=bfeec91efbb94961'] # Calm

audio_features = spotify.audio_features(input_tracks)
audio_features = np.array(audio_features)
audio_features_np = np.empty((3,11))
i = 0
relevant_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for i in range(len(audio_features)):
    for j in range(len(relevant_features)):
        audio_features_np[i][j] = audio_features[i][relevant_features[j]]
        j = j + 1
    i = i + 1

# Normalize audio features
StandardScaler().fit(audio_features_np)
print("AF input: ", audio_features_np)
kmeans = KMeans(n_clusters=3, random_state= 0).fit(audio_features_np)
print("Labels: ", kmeans.labels_)
print("Centers: ", kmeans.cluster_centers_)

# Check distance between centers of clusters. If under a threshold do KMeans again with one less cluster.
print(distance.euclidean(kmeans.cluster_centers_[0], kmeans.cluster_centers_[2]))

track_features_data = pd.read_csv('D:\Skola\ML\\tracks_features.csv')
track_features_data = track_features_data.loc[:, ['id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' ]]
print(track_features_data)
