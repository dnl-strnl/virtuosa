from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from virtuosa.play.library import Library, normalize


Test = Callable[[Dict], bool]

NAMES = {'type': 'array', 'items': {'type': 'string'}}
INTEGERS = {'type': 'array', 'items': {'type': 'integer'}}
INTEGER = {'type': 'integer'}
NUMBER = {'type': 'number'}

def song(track: Dict) -> Tuple[str, str]:
    """What makes two files the same song, whichever release they are on"""
    return normalize(track['artist']), normalize(track['title'])

@dataclass
class Selection:
    """What the filters of a tool call came to in the library"""
    tests: List[Test] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    # the tags found for each artist asked for, so that picks can take turns.
    artist_groups: List[set] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    suggestions: List[Dict] = field(default_factory=list)
    # reasons the call cannot be carried out as asked.
    gaps: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return ' '.join(filter(None, self.labels)).strip()

    def matches(self, track: Dict) -> bool:
        return all(test(track) for test in self.tests)

    def missing(self, library: Library, kind: str, what: str, heard: str) -> None:
        self.notes.append(f"No {what} like “{heard}” in your library.")
        if options := library.suggest(kind, heard):
            self.suggestions.append({'field': kind, 'heard': heard, 'options': options})

def artists(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    groups, names = [], []
    for artist in given['artists']:
        if found := library.find_artists(artist):
            groups.append(set(found))
            names.append(found[0])
        else:
            selection.missing(library, 'artists', 'artist', artist)
    selection.artist_groups += groups
    wanted = set().union(*groups)
    return (lambda t: t['artist'] in wanted), f"by {', '.join(names)}" if names else ''

def albums(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    wanted = set()
    for album in given['albums']:
        if found := library.find_albums(album):
            wanted.update(found)
        else:
            selection.missing(library, 'albums', 'album', album)
    return (lambda t: t['album'] in wanted), f"from {', '.join(sorted(wanted))}" if wanted else ''

def album_songs(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    """The songs on the named albums, on whichever release a track has them"""
    wanted = set().union(*(library.find_albums(album) for album in given['albums']))
    songs = {song(t) for t in library.tracks.values() if t['album'] in wanted}
    return (lambda t: song(t) in songs), f"from {', '.join(sorted(wanted))}" if wanted else ''

def titles(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    wanted, names = set(), []
    for title in given['titles']:
        if found := library.find_titles(title):
            wanted.update(found)
            names.append(library.tracks[found[0]]['title'])
        else:
            selection.missing(library, 'titles', 'track', title)
    return (lambda t: t['id'] in wanted), f"called {', '.join(names)}" if names else ''

def genres(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    wanted, names = set(), []
    for genre in given['genres']:
        if found := library.find_genres(genre):
            wanted.update(normalize(tag) for tag in found)
            names.append(found[0])
        else:
            selection.notes.append(f"No genre like “{genre}” in your library.")
    return (lambda t: t['genre_key'] in wanted), f"tagged {', '.join(names)}" if names else ''

def years(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    since, until = given.get('year_from') or 0, given.get('year_to') or 0
    if since and until:
        label = f"from {since}-{until}"
    else:
        label = f"since {since}" if since else f"up to {until}"
    return (lambda t: bool(t['year']) and since <= t['year'] <= (until or 9999)), label

def minutes(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    shortest, longest = given.get('min_minutes') or 0, given.get('max_minutes') or 0
    if shortest and longest:
        label = f"of {shortest:g}-{longest:g} min"
    else:
        label = f"over {shortest:g} min" if shortest else f"under {longest:g} min"
    return (lambda t: shortest * 60 <= t['duration'] <= (longest or float('inf')) * 60), label

def title_words(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    # padded with spaces so that a word does not match inside a longer one.
    words = [f" {normalize(word)} " for word in given['title_words']]
    test = lambda t: any(word in f" {normalize(t['title'])} " for word in words)
    return test, f"with {', '.join(given['title_words'])} in the title"

def album_track_numbers(library: Library, given: Dict, selection: Selection) -> Tuple[Test, str]:
    numbers = set(given['album_track_numbers'])
    return (lambda t: t['track_number'] in numbers), f"that are track {', '.join(map(str, sorted(numbers)))} of their album"

@dataclass
class Filter:
    """One way of narrowing tracks down, or a stub of it when it has no `build` yet"""
    name: str
    properties: Dict[str, Dict]
    description: str
    build: Optional[Callable[[Library, Dict, Selection], Tuple[Test, str]]] = None
    # used instead of `build` when the filter rules tracks out.
    exclude: Optional[Callable[[Library, Dict, Selection], Tuple[Test, str]]] = None

    def given(self, filters: Dict) -> Dict:
        return {name: filters[name] for name in self.properties if filters.get(name)}

FILTERS = [
    Filter('artist', {'artists': NAMES}, "names of artists.", artists),
    # ruling an album out takes the same song on another release with it.
    Filter('album', {'albums': NAMES}, "names of albums.", albums, exclude=album_songs),
    Filter('title', {'titles': NAMES}, "names of songs.", titles),
    Filter('genre', {'genres': NAMES}, "genres, spelled as in the genres in the library.", genres),
    Filter('year', {'year_from': INTEGER, 'year_to': INTEGER}, "the years a track may be from.", years),
    Filter(
        'length', {'min_minutes': NUMBER, 'max_minutes': NUMBER},
        'how long a track is ("under four minutes" is max_minutes 4).', minutes,
    ),
    Filter(
        'words in the title', {'title_words': NAMES},
        'words that a title has in it ("songs with rain in the name").', title_words,
    ),
    Filter(
        'album track number', {'album_track_numbers': INTEGERS},
        "where a track sits on its own album, where 1 is the album's opening track.", album_track_numbers,
    ),
    Filter(
        'tempo', {'min_bpm': INTEGER, 'max_bpm': INTEGER},
        "tempo in beats per minute, only when the request gives a number.",
    ),
    Filter('play count', {'min_plays': INTEGER, 'max_plays': INTEGER}, "how many times a track has been played."),
    Filter('date added', {'added_within_days': INTEGER}, "how recently a track was added to the library."),
    Filter('rating', {'min_rating': INTEGER}, "the user's rating of a track, from 1 to 5."),
    Filter('lyrics', {'lyrics': NAMES}, "words or subjects that the lyrics have in them."),
    Filter('similarity', {'similar_to': NAMES}, "songs or artists that a track should sound like."),
]
PROPERTIES = {name: schema for each in FILTERS for name, schema in each.properties.items()}

@dataclass
class Order:
    """One way of putting tracks in order, or a stub of it when it has no `key` yet"""
    name: str
    description: str
    key: Optional[Callable[[Dict], object]] = None

ORDERS = [
    Order('title', "alphabetical.", lambda t: normalize(t['title'])),
    Order(
        'artist', "by artist, with each artist's albums in their own order.",
        lambda t: (normalize(t['artist']), normalize(t['album']), t['disc_number'], t['track_number']),
    ),
    Order('album', "the album's own track order.", lambda t: (normalize(t['album']), t['disc_number'], t['track_number'])),
    Order('year', "oldest first.", lambda t: t['year'] or 0),
    Order('duration', "shortest first.", lambda t: t['duration']),
    Order('genre', "by genre.", lambda t: t['genre_key']),
    Order('tempo', "slowest first."),
    Order('plays', "least played first."),
    Order('added', "longest in the library first."),
    Order('rating', "lowest rated first."),
]
ORDER_KEYS = {order.name: order.key for order in ORDERS if order.key}

def guide(entries: List) -> str:
    """The filters or orders as lines that explain them to the model"""
    names = lambda entry: ', '.join(entry.properties) if isinstance(entry, Filter) else entry.name
    return '\n'.join(f"- {names(entry)}: {entry.description}" for entry in entries)

def select(library: Library, where: Dict, exclude: Dict) -> Selection:
    """Work out the `where` and `exclude` of a tool call against the library"""
    selection = Selection()
    selection.gaps += [f"There is no filter called {name}." for name in {**where, **exclude} if name not in PROPERTIES]
    selection.gaps += [
        f"Can't filter by {each.name} yet." for each in FILTERS
        if each.build is None and (each.given(where) or each.given(exclude))
    ]
    built = [each for each in FILTERS if each.build]

    for each in built:
        if given := each.given(where):
            test, label = each.build(library, given, selection)
            selection.tests.append(test)
            selection.labels.append(label)

    # a name that is not in the library rules nothing out, so it is not reported.
    unreported, unwanted = Selection(), []
    for each in built:
        if given := each.given(exclude):
            test, label = (each.exclude or each.build)(library, given, unreported)
            selection.tests.append(lambda t, test=test: not test(t))
            unwanted.append(label)
    if unwanted := list(filter(None, unwanted)):
        selection.labels.append(f"but not {', '.join(unwanted)}")
    return selection
