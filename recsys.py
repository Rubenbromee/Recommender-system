import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import configparser

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

gm = spotify.track('https://open.spotify.com/track/5CaXxLM568tBh1PwhXdciZ?si=3515cd0387f84792')
songs = ['https://open.spotify.com/track/5CaXxLM568tBh1PwhXdciZ?si=3515cd0387f84792', 'https://open.spotify.com/track/2aHlRZIGUFThu3eQePm6yI?si=8612644bf9154a3c']
af = spotify.audio_features(songs)
print(gm['name'])
print(gm['explicit'])
print(gm['popularity'])
print(af[1]['acousticness'])

