import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import configparser
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pw

config = configparser.ConfigParser()
config.read('config.cfg')
client_id = config.get('SPOTIFY', 'CLIENT_ID')
client_secret = config.get('SPOTIFY', 'CLIENT_SECRET')

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))

# k_west = 'spotify:artist:5K4W6rqBFWDnAN6FQUkS6x'
# res = spotify.artist_albums(k_west, album_type='album')
# print(res)

# albums = res['items']
# while res['next']:
#     res = spotify.next(res)
#     albums.extend(res['items'])

# for album in albums:
#     print(album['name'])

# gm = spotify.track('https://open.spotify.com/track/5CaXxLM568tBh1PwhXdciZ?si=3515cd0387f84792')
# songs = ['https://open.spotify.com/track/5CaXxLM568tBh1PwhXdciZ?si=3515cd0387f84792', 'https://open.spotify.com/track/2aHlRZIGUFThu3eQePm6yI?si=8612644bf9154a3c']
# af = spotify.audio_features(songs)
# print(gm['name'])
# print(gm['explicit'])
# print(gm['popularity'])
# print(af[1]['acousticness'])

input_tracks = ['https://open.spotify.com/track/1YQWosTIljIvxAgHWTp7KP?si=b36a46f624694cb3', 
'https://open.spotify.com/track/4dzBqc5t2GWJKpGqUoTbrU?si=79152cc8d9a246aa', 
'https://open.spotify.com/track/6MTtOV150uwitRZz4anI2h?si=d57bb82ca2f6425a']

audio_features = spotify.audio_features(input_tracks)
# print("Audio feature properties")
# print(len(audio_features[0]))
# for af in audio_features[0]:
#     print(af)


# print("Audio analysis properties")
# audio_analysis_0 = spotify.audio_analysis(input_tracks[0])
# print(len(audio_analysis_0))
# for aa in audio_analysis_0:
#     print(aa)

# print(audio_analysis_0['tatums'])

audio_features = np.array(audio_features)
audio_features_np = np.empty((3,11))

i = 0
relevant_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for i in range(len(audio_features)):
    for j in range(len(relevant_features)):
        audio_features_np[i][j] = audio_features[i][relevant_features[j]]
        j = j + 1
    i = i + 1

# MUD RACING:
# print(audio_features_np)
# avg_track = np.sum(audio_features_np, axis=0)
# print(avg_track)
# # Mud racing
# # i = 0
# # for i in range(len(avg_track)):
# #     if i == 3:
# # avg_track = avg_track / 3
# print(avg_track)

track_features_data = pd.read_csv('D:\Skola\ML\\tracks_features.csv')
track_features_data = track_features_data.loc[:, ['id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' ]]
print(track_features_data)

# Cosine similarity between each feature in each song

# Take column for one feature from feature data
# Extract the same feature from the inmput tracks
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
# Create an n x m matrix of cosine similarities for a feature where n is the number of songs in our dataset and m is the number of input songs
# Take the largest row sum as the most similar song for that feature
# Mud racing

track_features_danceability = track_features_data.loc[:, 'danceability']
track_features_danceability = np.array(track_features_danceability)
cs = pw.cosine_similarity(track_features_danceability.reshape(-1,1), audio_features_np[:][0].reshape(-1,1))
print(cs)
