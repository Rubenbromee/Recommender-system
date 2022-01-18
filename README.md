# Recommender-system

A recommender system for Spotify using the library Spotipy.
In order to use, you need a database of songs gathered using the Spotify API. We used https://www.kaggle.com/rodolfofigueroa/spotify-12m-songs, but could not include it due to size limitations.
This file should be pointed to by the track_features_data variable, initialized at line 51.
The input songs can be changed to any song by going into Spotify, right clicking on a song, then clicking "Copy Song Link", and pasting that link into the input_tracks array. Note that this array should be size 5.
