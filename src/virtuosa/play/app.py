from flask import Flask, Response, abort, jsonify, render_template, request, send_file
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import whisper

from virtuosa.play.curator import PlaylistCurator
from virtuosa.play.library import Library

app = Flask(__name__, static_folder='static', template_folder='templates')

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

music_library = Path("~/Music/Music/Media.localized/Music").expanduser()

library = Library(music_library, db_path="virtuosa.db")
library.scan()

curator = PlaylistCurator(library, model_name=os.environ.get('VIRTUOSA_MODEL', 'qwen3:8b'))
threading.Thread(target=curator.warm_up, daemon=True).start()

whisper_model = None
whisper_lock = threading.Lock()

def transcribe(recording, playlist=None):
    global whisper_model
    # whisper keeps only the end of a long prompt, so the playlist's artists go last.
    artists = library.top_artists()
    if playlist:
        artists += list(dict.fromkeys(track['artist'] for track in playlist['tracks']))
    with whisper_lock:
        if whisper_model is None:
            whisper_model = whisper.load_model("base.en")
        result = whisper_model.transcribe(
            recording,
            fp16=False,
            initial_prompt=f"A request for a music playlist. Artists: {', '.join(artists)}.",
        )
    return result["text"].strip()

@app.route('/')
def index():
    return render_template('index.html', track_count=len(library.tracks), artist_count=len(library.artists))

@app.route('/api/playlists')
def get_playlists():
    return jsonify(library.playlists())

@app.route('/api/playlists/<int:playlist_id>', methods=['GET', 'DELETE'])
def playlist(playlist_id):
    if request.method == 'DELETE':
        library.delete_playlist(playlist_id)
        return jsonify(dict(deleted=playlist_id))
    if (found := library.playlist(playlist_id)) is None:
        abort(404)
    return jsonify(found)

@app.route('/api/command', methods=['POST'])
def command():
    """Create or edit a playlist from a request, streaming one JSON event per line"""
    text = request.form.get('text', '').strip()
    playlist_id = request.form.get('playlist_id', type=int)

    recording = None
    if 'audio' in request.files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            request.files['audio'].save(temp_audio)
            recording = temp_audio.name
    elif not text:
        return jsonify(dict(error='No audio or text provided.')), 400

    def events():
        try:
            existing = library.playlist(playlist_id) if playlist_id else None

            prompt = transcribe(recording, existing) if recording else text
            log.info(f'command: {prompt}')
            if not prompt:
                raise ValueError("Didn't catch that, try again.")
            yield dict(event='heard', text=prompt)

            name = existing['name'] if existing else ''
            track_ids = [track['id'] for track in existing['tracks']] if existing else []
            commands = existing['commands'] if existing else []

            for event in curator.edit(prompt, name, track_ids):
                if event['event'] == 'op':
                    yield dict(event, tracks=library.public(event['track_ids']))
                else:
                    name, track_ids = event['name'], event['track_ids']

            if not existing and not track_ids:
                raise ValueError("Couldn't find anything in your library for that.")

            saved_id = library.save_playlist(existing and existing['id'], name, track_ids, commands + [prompt])
            yield dict(event='done', playlist=library.playlist(saved_id))
        except ValueError as command_error:
            yield dict(event='error', error=str(command_error))
        except Exception as command_error:
            log.exception(f"{command_error=}")
            yield dict(event='error', error=str(command_error))
        finally:
            if recording:
                os.remove(recording)

    return Response((json.dumps(event) + '\n' for event in events()), mimetype='application/x-ndjson')

@app.route('/api/correct', methods=['POST'])
def correct():
    """Redo an operation with the name the user picked, and remember the choice"""
    data = request.get_json()
    field, heard, choice = data.get('field'), data.get('heard'), data.get('choice')
    if field not in ('artists', 'albums', 'titles') or not heard or not choice:
        return jsonify(dict(error='Nothing to correct.')), 400
    library.remember(field, heard, choice)

    existing = library.playlist(data['playlist_id']) if data.get('playlist_id') else None
    track_ids = [track['id'] for track in existing['tracks']] if existing else []
    operation = data.get('operation', {})
    result = curator.apply({**operation, 'where': {**operation.get('where', {}), field: [choice]}}, track_ids)
    if not existing and not result['track_ids']:
        return jsonify(dict(error="Couldn't find anything in your library for that.")), 404

    saved_id = library.save_playlist(
        existing and existing['id'],
        existing['name'] if existing else choice,
        result['track_ids'],
        (existing['commands'] if existing else []) + [f"{heard} -> {choice}"],
    )
    return jsonify(dict(result, playlist=library.playlist(saved_id)))

@app.route('/stream/<int:track_id>')
def stream_audio(track_id):
    if (track := library.tracks.get(track_id)) is None:
        abort(404)
    return send_file(track['filepath'], conditional=True)

if __name__ == '__main__':
    app.run(port=5889, debug=True)
