from collections import Counter
import difflib
from functools import lru_cache
import json
import logging
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from pathlib import Path
import re
import sqlite3
import tqdm
from typing import Dict, List, Optional
import unicodedata


log = logging.getLogger(__name__)

TAG_COLUMNS = (
    'title', 'artist', 'album', 'album_artist', 'genre',
    'year', 'track_number', 'disc_number', 'duration',
)

def normalize(text: str) -> str:
    """Lowercase, without accents or punctuation: 'Café Olé!' becomes 'cafe ole'"""
    text = unicodedata.normalize('NFKD', (text or '').casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(re.findall(r'\w+', text))

# consonants that sound alike share a digit, as in soundex.
SOUNDS = {
    **dict.fromkeys('bfpv', '1'), **dict.fromkeys('cgjkqsxz', '2'), **dict.fromkeys('dt', '3'),
    'l': '4', **dict.fromkeys('mn', '5'), 'r': '6',
}

@lru_cache(maxsize=None)
def phonetic(text: str) -> str:
    """Code for how a name sounds: 'Catherine' and 'Kathryn' are both '2365'"""
    compact = normalize(text).replace(' ', '')
    code = 'a' if compact[:1] in tuple('aeiouy') else ''
    for char in compact:
        sound = SOUNDS.get(char, char if char.isdigit() else '')
        if sound and not code.endswith(sound):
            code += sound
        elif not sound and char not in 'hw':
            code += ' '  # a vowel lets the same sound repeat.
    return code.replace(' ', '')

def leading_int(value: str) -> Optional[int]:
    match = re.match(r'\d+', value or '')
    return int(match.group()) if match else None

def read_tags(filepath: Path) -> Dict:
    audio = MP3(filepath, ID3=EasyID3)
    return {
        'title': audio.get('title', [filepath.stem])[0],
        'artist': audio.get('artist', ['Unknown'])[0],
        'album': audio.get('album', ['Unknown'])[0],
        'album_artist': audio.get('albumartist', [''])[0],
        'genre': audio.get('genre', [''])[0],
        'year': leading_int(audio.get('date', [''])[0]),
        'track_number': leading_int(audio.get('tracknumber', ['0'])[0]) or 0,
        'disc_number': leading_int(audio.get('discnumber', ['0'])[0]) or 0,
        'duration': int(audio.info.length),
    }

class Library:
    def __init__(self, music_dir: Path, db_path: str = "virtuosa.db"):
        self.music_dir = Path(music_dir)
        self.db_path = db_path
        self.tracks: Dict[int, Dict] = {}
        self.artists: Dict[str, set] = {}
        self.albums: Dict[str, set] = {}
        self.titles: Dict[str, set] = {}
        self.genres: List[str] = []
        self.aliases: Dict[tuple, str] = {}

        with self.connect() as db:
            db.execute(
                "create table if not exists tracks ("
                "id integer primary key, filepath text unique, mtime real, "
                f"{', '.join(TAG_COLUMNS)})"
            )
            db.execute(
                "create table if not exists playlists ("
                "id integer primary key, name text, track_ids text, commands text)"
            )
            # a misheard name and the library name the user picked for it.
            db.execute(
                "create table if not exists aliases (kind text, heard text, actual text, primary key (kind, heard))"
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def scan(self) -> None:
        """Sync the database with disk, reading tags only for new or changed files"""
        files = list(self.music_dir.rglob("*.mp3"))
        with self.connect() as db:
            known = {row['filepath']: row for row in db.execute("select id, filepath, mtime from tracks")}

            for filepath in tqdm.tqdm(files):
                mtime = filepath.stat().st_mtime
                if (row := known.pop(str(filepath), None)) and row['mtime'] == mtime:
                    continue
                try:
                    tags = read_tags(filepath)
                except Exception as load_audio_file_error:
                    log.error(f"{filepath}: {load_audio_file_error=}")
                    continue
                db.execute(
                    f"insert into tracks (filepath, mtime, {', '.join(TAG_COLUMNS)}) "
                    f"values (?, ?, {', '.join('?' * len(TAG_COLUMNS))}) "
                    "on conflict(filepath) do update set mtime = excluded.mtime, "
                    + ', '.join(f"{column} = excluded.{column}" for column in TAG_COLUMNS),
                    [str(filepath), mtime, *(tags[column] for column in TAG_COLUMNS)],
                )

            # rows left over have no file, unless the whole library is missing.
            if files:
                db.executemany("delete from tracks where id = ?", [(row['id'],) for row in known.values()])
            elif known:
                log.warning(f"no mp3 files under {self.music_dir}, keeping the previous scan.")

        self.load()

    def load(self) -> None:
        """Read the tracks into memory and index the names we look things up by"""
        with self.connect() as db:
            self.tracks = {row['id']: dict(row) for row in db.execute("select * from tracks")}
            self.aliases = {
                (row['kind'], row['heard']): row['actual'] for row in db.execute("select * from aliases")
            }

        self.artists, self.albums, self.titles = {}, {}, {}
        for track in self.tracks.values():
            self.artists.setdefault(normalize(track['artist']), set()).add(track['artist'])
            self.albums.setdefault(normalize(track['album']), set()).add(track['album'])
            self.titles.setdefault(normalize(track['title']), set()).add(track['id'])
            track['genre_key'] = normalize(track['genre'])

        genres = Counter(track['genre'] for track in self.tracks.values() if track['genre_key'])
        self.genres = [genre for genre, _ in genres.most_common()]

    def top_artists(self, n: int = 40) -> List[str]:
        counts = Counter(track['artist'] for track in self.tracks.values())
        return [artist for artist, _ in counts.most_common(n)]

    def index(self, kind: str) -> Dict[str, set]:
        return {'artists': self.artists, 'albums': self.albums, 'titles': self.titles}[kind]

    def display(self, kind: str, key: str) -> str:
        names = self.index(kind)[key]
        if kind == 'titles':
            names = {self.tracks[track_id]['title'] for track_id in names}
        return min(names, key=len)

    def closest(self, kind: str, name: str, cutoff: float) -> Optional[str]:
        """The library's normalized name closest to a possibly misheard one"""
        keys = self.index(kind)
        if not (key := normalize(name)):
            return None
        if key in keys:
            return key
        if (alias := self.aliases.get((kind, key))) in keys:
            return alias
        # ignore spaces, since a name can be heard as one word or two.
        compact = {k.replace(' ', ''): k for k in keys}
        matches = difflib.get_close_matches(key.replace(' ', ''), compact, n=1, cutoff=cutoff)
        return compact[matches[0]] if matches else None

    def suggest(self, kind: str, name: str, n: int = 3) -> List[str]:
        """Library names spelled or pronounced like `name`, best first"""
        compact, code = normalize(name).replace(' ', ''), phonetic(name)
        ratio = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0
        scored = []
        for key in self.index(kind):
            score = (ratio(compact, key.replace(' ', '')) + ratio(code, phonetic(key))) / 2
            if score >= 0.62:
                scored.append((score, key))
        return [self.display(kind, key) for _, key in sorted(scored, reverse=True)[:n]]

    def remember(self, kind: str, heard: str, actual: str) -> None:
        """Keep a correction so the misheard name finds the right one next time"""
        heard, actual = normalize(heard), normalize(actual)
        if not heard or actual not in self.index(kind):
            return
        with self.connect() as db:
            db.execute("insert or replace into aliases (kind, heard, actual) values (?, ?, ?)", [kind, heard, actual])
        self.aliases[(kind, heard)] = actual

    def find_artists(self, name: str) -> List[str]:
        """Artist tags matching a name, with collaborations like 'X & Y'"""
        if not (key := self.closest('artists', name, cutoff=0.75)):
            return []
        found = [self.display('artists', key), *self.artists[key]]
        if len(key) >= 4:
            for other, tags in self.artists.items():
                if other != key and f" {key} " in f" {other} ":
                    found.extend(tags)
        return found

    def sounds_like(self, text: str) -> List[str]:
        """Artists that a run of words in the text may be a mishearing of"""
        words = normalize(text).split()
        found, used = {}, set()
        # longest runs first, so a full name beats a match on one of its words.
        for size in range(4, 0, -1):
            for i in range(len(words) - size + 1):
                span = set(range(i, i + size))
                phrase = ' '.join(words[i:i + size])
                if len(phrase) < 5 or span & used:
                    continue
                if key := self.closest('artists', phrase, cutoff=0.75):
                    found[key] = self.display('artists', key)
                    used |= span
        return list(found.values())

    def find_albums(self, name: str) -> List[str]:
        key = self.closest('albums', name, cutoff=0.8)
        return list(self.albums[key]) if key else []

    def find_titles(self, name: str) -> List[int]:
        key = self.closest('titles', name, cutoff=0.8)
        return list(self.titles[key]) if key else []

    def find_genres(self, name: str) -> List[str]:
        # the most common spelling of a genre is the one to show.
        keys = {normalize(genre): genre for genre in reversed(self.genres)}
        matches = difflib.get_close_matches(normalize(name), keys, n=1, cutoff=0.8)
        return [keys[match] for match in matches]

    def public(self, track_ids: List[int]) -> List[Dict]:
        """Tracks as sent to the browser, skipping any no longer in the library"""
        fields = ('id', 'title', 'artist', 'album', 'genre', 'year', 'duration')
        return [
            {field: self.tracks[track_id][field] for field in fields}
            for track_id in track_ids if track_id in self.tracks
        ]

    def playlists(self) -> List[Dict]:
        with self.connect() as db:
            rows = db.execute("select * from playlists order by id desc").fetchall()
        return [
            {'id': row['id'], 'name': row['name'], 'count': len(json.loads(row['track_ids']))}
            for row in rows
        ]

    def playlist(self, playlist_id: int) -> Optional[Dict]:
        with self.connect() as db:
            row = db.execute("select * from playlists where id = ?", [playlist_id]).fetchone()
        if row is None:
            return None
        return {
            'id': row['id'],
            'name': row['name'],
            'commands': json.loads(row['commands']),
            'tracks': self.public(json.loads(row['track_ids'])),
        }

    def save_playlist(self, playlist_id: Optional[int], name: str, track_ids: List[int], commands: List[str]) -> int:
        values = [name, json.dumps(track_ids), json.dumps(commands)]
        with self.connect() as db:
            if playlist_id is None:
                return db.execute(
                    "insert into playlists (name, track_ids, commands) values (?, ?, ?)", values
                ).lastrowid
            db.execute(
                "update playlists set name = ?, track_ids = ?, commands = ? where id = ?",
                [*values, playlist_id],
            )
            return playlist_id

    def delete_playlist(self, playlist_id: int) -> None:
        with self.connect() as db:
            db.execute("delete from playlists where id = ?", [playlist_id])
