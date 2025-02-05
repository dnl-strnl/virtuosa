from flask import Flask, jsonify, render_template, request, send_file
import json
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
import ollama
import os
from pathlib import Path
import pprint
import tempfile
import tqdm
import whisper

from virtuosa.play.chain import MusicLibraryChain

app = Flask(__name__, static_folder='static', template_folder='templates')

music_library = Path("~/Music/Music/Media.localized/Music").expanduser()
playlists_file = 'playlists.json'
whisper_model = whisper.load_model("tiny")

def scan_music_library(files, verbose=False):
    music_data = []

    for f in tqdm.tqdm(files):
        try:
            audio = MP3(filepath := Path(f), ID3=EasyID3)

            track_info = {
                'filepath': filepath,
                'title': audio.get('title', [filepath.stem])[0],
                'artist': audio.get('artist', ['Unknown'])[0],
                'album': audio.get('album', ['Unknown'])[0],
                'album_artist': audio.get('albumartist', ['Unknown'])[0],
                'composer': audio.get('composer', ['Unknown'])[0],
                'genre': audio.get('genre', ['Unknown'])[0],
                'year': audio.get('date', ['Unknown'])[0],
                'track_number': audio.get('tracknumber', ['0'])[0],
                'disc_number': audio.get('discnumber', ['0'])[0],
                'bpm': audio.get('bpm', ['Unknown'])[0],
                'duration': int(audio.info.length),
                'bitrate': audio.info.bitrate // 1000, # in kbps
            }

            if verbose:
                pprint.pprint(track_info, indent=4)
            music_data.append(track_info)
        except Exception as load_audio_file_error:
            log.error(f"{load_audio_file_error=}")

    return music_data

music_chain = MusicLibraryChain()
files = list(music_library.rglob("*.mp3"))
music_chain.initialize_chain(library_data := scan_music_library(files))

@app.route('/')
def index():
    playlists = []
    if os.path.exists(playlists_file):
        with open(playlists_file, 'r') as f:
            playlists = json.load(f)

    music_files = list(music_library.rglob("*.mp3"))
    return render_template('index.html', playlists=playlists, library=library_data)

@app.route('/api/library')
def get_library():
    return jsonify(library_data)

@app.route('/api/playlists')
def get_playlists():
    if os.path.exists(playlists_file):
        with open(playlists_file, 'r') as f:
            playlists = json.load(f)
        return jsonify(playlists)
    return jsonify([])

@app.route('/api/create-playlist/voice', methods=['POST'])
def create_playlist_from_voice():
    if 'audio' not in request.files:
        return jsonify(dict(error='No audio file provided.')), 400

    audio_file = request.files['audio']

    # save the uploaded audio file temporarily.
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        audio_file.save(temp_audio.name)

        # transcribe audio using Whisper.
        result = whisper_model.transcribe(temp_audio.name)
        prompt = result["text"]

        try:
            # create a playlist from the user collection + input prompt.
            playlist = music_chain.generate_playlist(prompt)

            # save playlist metadata.
            playlists = []
            if os.path.exists(playlists_file):
                with open(playlists_file, 'r') as f:
                    playlists = json.load(f)

            playlists.append(playlist)
            with open(playlists_file, 'w') as f:
                json.dump(playlists, f)

            return jsonify(playlist)
        except Exception as create_playlist_error:
            return jsonify(dict(error=create_playlist_error)), 500

@app.route('/stream/<path:filename>')
def stream_audio(filename):
    return send_file(os.path.join(music_library, filename))

if __name__ == '__main__':
    app.run(debug=True)
