from collections import Counter
from dataclasses import dataclass, field
import json
import random
import re
from typing import Callable, Dict, List, Optional, Tuple

from virtuosa.play.filters import ORDERS, ORDER_KEYS, PROPERTIES, Selection, Test, album_songs, select, song
from virtuosa.play.library import Library, normalize


DEFAULT_COUNT = 10
PER_ARTIST_COUNT = 5
# a longer run at `to` is a block of tracks, not an opener or closer to pin.
PINNED_RUN = 3
# rows of a lookup that the model is shown.
SHOWN = 20

# an exclusion counts only if the call's `request` has one of these.
NEGATIVES = re.compile(
    r"\b(not|no|nothing|none|never|without|except|excluding|minus|but|other than|instead of|leave out)\b|n['’]t",
    re.IGNORECASE,
)
# and numbers for an add to land on, only if it has one of these.
NUMBERS = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|first|second|third|fourth"
    r"|fifth|sixth|seventh|eighth|ninth|tenth|last|prime|odd|even|numbers?|slots?|positions?)\b",
    re.IGNORECASE,
)
# and a new name for a playlist that has one, only if it has one of these.
NAMING = re.compile(r"\b(name|rename|call|title)\b", re.IGNORECASE)

GROUPS = {
    'artist': lambda t: normalize(t['artist']),
    'album': lambda t: normalize(t['album']),
    'genre': lambda t: t['genre_key'],
}
FIELDS = {
    'artists': lambda t: t['artist'],
    'albums': lambda t: t['album'],
    'genres': lambda t: t['genre'],
    'years': lambda t: t['year'],
}

def plural(n: int) -> str:
    return f"{n} track" if n == 1 else f"{n} tracks"

def insert(track_ids: List[int], new_ids: List[int], to: int) -> List[int]:
    """Insert `new_ids` so the first lands at number `to`, or at the end"""
    at = to - 1 if 1 <= to <= len(track_ids) else len(track_ids)
    return track_ids[:at] + new_ids + track_ids[at:]

def place(track_ids: List[int], new_ids: List[int], numbers: List[int]) -> List[int]:
    """Insert `new_ids` on the given numbers, with any left over at the end"""
    landing = set(sorted(set(numbers))[:len(new_ids)])
    old, new = iter(track_ids), list(new_ids)
    merged = []
    for number in range(1, len(track_ids) + len(new_ids) + 1):
        if number in landing and new:
            merged.append(new.pop(0))
        elif (track_id := next(old, None)) is not None:
            merged.append(track_id)
        else:
            merged.append(new.pop(0))
    return merged

def rearrange(track_ids: List[int], slots: List[int], ordered: List[int]) -> List[int]:
    """Put `ordered` back into `slots`, leaving every other track where it is"""
    track_ids = list(track_ids)
    for slot, track_id in zip(slots, ordered):
        track_ids[slot] = track_id
    return track_ids

def placed(call: Dict, before: List[int], after: List[int]) -> Dict[int, int]:
    """The numbers an add was asked to put its tracks on, by track id"""
    to, numbers = call.get('to') or 0, call.get('positions')
    if call.get('action') != 'add' or not (to or numbers):
        return {}
    old = set(before)
    new = [(number, track_id) for number, track_id in enumerate(after, start=1) if track_id not in old]
    if not numbers and len(new) > PINNED_RUN:
        return {}
    # a closer counts back from the end, so the last track is -1.
    if to < 0 and not numbers:
        return {track_id: number - len(after) - 1 for number, track_id in new}
    return {track_id: number for number, track_id in new}

def hold(track_ids: List[int], pins: Dict[int, int]) -> List[int]:
    """Put pinned tracks back on the numbers they were asked for"""
    pinned = sorted((number, track_id) for track_id, number in pins.items() if track_id in track_ids)
    held = [track_id for track_id in track_ids if track_id not in pins]
    for number, track_id in pinned:
        if number > 0:
            held.insert(min(number - 1, len(held)), track_id)
    return held + [track_id for number, track_id in pinned if number < 0]

def pick(pool: List[Dict], artist_groups: List[set], count: int) -> List[Dict]:
    """Choose `count` tracks at random, taking turns between the artists"""
    random.shuffle(pool)
    if len(artist_groups) < 2:
        return pool[:count]
    turns = [[t for t in pool if t['artist'] in group] for group in artist_groups]
    picked, seen = [], set()
    while len(picked) < count and any(turns):
        for turn in turns:
            if turn and len(picked) < count and (track := turn.pop())['id'] not in seen:
                seen.add(track['id'])
                picked.append(track)
    return picked

@dataclass
class Candidate:
    """The playlist that a request is being carried out on"""
    name: str
    track_ids: List[int]
    # a track asked for at a number stays there for the rest of the request.
    pins: Dict[int, int] = field(default_factory=dict)

class Step:
    """One tool call on the candidate, with its arguments worked out"""

    def __init__(self, library: Library, candidate: Candidate, call: Dict, selection: Selection):
        self.library, self.candidate, self.call, self.selection = library, candidate, call, selection
        self.tracks = library.tracks
        self.filters = call.get('where') or {}
        self.count = max(call.get('count') or 0, 0)
        self.to = call.get('to') or 0
        self.order = ORDER_KEYS.get(call.get('by'))
        self.descending = bool(call.get('descending'))
        self.per = call.get('per') if call.get('per') in GROUPS else 'artist'
        self.pinned = set(candidate.pins)
        # what the model is told, when that is more than the summary.
        self.reply = ''

        track_ids = candidate.track_ids
        self.place = f"at number {self.to}" if 1 <= self.to <= len(track_ids) else ''
        positions = list(dict.fromkeys(call.get('positions') or []))
        # on an add, positions are where the new tracks land, not existing tracks.
        self.landing = sorted(p for p in positions if p >= 1) if call.get('action') == 'add' else []
        if self.landing:
            positions = []
            self.place = f"at number{'s' if len(self.landing) > 1 else ''} {', '.join(map(str, self.landing))}"

        if off_the_end := [str(p) for p in positions if not 1 <= p <= len(track_ids)]:
            selection.notes.append(f"There is no track {', '.join(off_the_end)} in the playlist.")
        self.positions = positions
        self.slots = sorted(p - 1 for p in positions if 1 <= p <= len(track_ids)) if positions else range(len(track_ids))
        self.chosen = [i for i in self.slots if selection.matches(self.tracks[track_ids[i]])]
        self.targeted = bool(selection.tests or positions)

    def what(self, n: int) -> str:
        """The tracks a call worked on, as its summary names them"""
        return f"{plural(n)} {self.selection.label}".strip() if self.targeted else "the playlist"

def add(step: Step, verb: str = "Added") -> str:
    candidate, selection, filters = step.candidate, step.selection, step.filters
    current = set(candidate.track_ids)
    pool = [t for t in step.tracks.values() if t['id'] not in current and selection.matches(t)]
    groups = selection.artist_groups
    default = PER_ARTIST_COUNT * len(groups) if len(groups) > 1 else DEFAULT_COUNT
    # naming a place asks for one track.
    wanted = step.count or len(step.landing) or (1 if step.place or step.to < 0 else default)
    # a named song or album comes whole, unless the request says how many.
    whole = bool(filters.get('titles') or filters.get('albums')) and not step.count
    if step.order:
        pool.sort(key=step.order, reverse=step.descending)
        picked = pool if whole else pool[:wanted]
    elif filters.get('titles') and not step.count:
        picked = pool
    elif filters.get('albums') and not step.count:
        picked = sorted(pool, key=lambda t: (t['album'], t['disc_number'], t['track_number']))
    else:
        picked = pick(pool, groups, wanted)
    if not picked:
        return "Found nothing to add."
    summary = ' '.join(filter(None, [verb, plural(len(picked)), selection.label, step.place]))
    new_ids = [t['id'] for t in picked]
    if step.landing:
        candidate.track_ids = place(candidate.track_ids, new_ids, step.landing)
    else:
        candidate.track_ids = insert(candidate.track_ids, new_ids, step.to)
    return f"{summary}."

def remove(step: Step) -> str:
    candidate, chosen = step.candidate, step.chosen
    if not step.targeted and not step.count:
        return "Wasn't sure what to remove."
    # no track is from the album, so match its songs on other releases.
    if not chosen and step.filters.get('albums'):
        on_album, _ = album_songs(step.library, step.filters, Selection())
        chosen = [i for i in step.slots if on_album(step.tracks[candidate.track_ids[i]])]
    doomed = set(chosen[-step.count:] if step.count else chosen)
    if not doomed:
        return "Found nothing to remove."
    summary = f"Removed {plural(len(doomed))} {step.selection.label}".strip()
    candidate.track_ids = [track_id for i, track_id in enumerate(candidate.track_ids) if i not in doomed]
    return f"{summary}."

def keep(step: Step) -> str:
    candidate, chosen = step.candidate, step.chosen
    if not step.targeted:
        return "Wasn't sure what to keep."
    # a misheard name should not wipe the playlist.
    if not chosen:
        return "Nothing in the playlist matched, so left it alone."
    summary = f"Kept {plural(len(chosen))} {step.selection.label}".strip()
    removed = len(candidate.track_ids) - len(chosen)
    candidate.track_ids = [candidate.track_ids[i] for i in chosen]
    return f"{summary}, removed {removed}."

def move(step: Step) -> str:
    candidate, chosen = step.candidate, step.chosen
    if not step.targeted:
        return "Wasn't sure what to move."
    if not chosen:
        return "Found nothing to move."
    moving = [candidate.track_ids[i] for i in chosen]
    staying = [track_id for i, track_id in enumerate(candidate.track_ids) if i not in set(chosen)]
    where = f"to number {step.to}" if step.place else "to the end"
    summary = ' '.join(filter(None, ["Moved", plural(len(moving)), step.selection.label, where]))
    candidate.track_ids = insert(staying, moving, step.to)
    return f"{summary}."

def swap(step: Step) -> str:
    if len(step.chosen) != 2:
        return "Need exactly two tracks to swap."
    a, b = step.chosen
    track_ids = step.candidate.track_ids
    step.candidate.track_ids = rearrange(track_ids, [a, b], [track_ids[b], track_ids[a]])
    return f"Swapped tracks {a + 1} and {b + 1}."

def resize(step: Step) -> str:
    candidate, selection, tracks, count = step.candidate, step.selection, step.tracks, step.count
    track_ids = candidate.track_ids
    if not count:
        return "Nothing to do."
    if count <= len(track_ids):
        # trim from the end, sparing pinned tracks.
        spare = [i for i, track_id in enumerate(track_ids) if track_id not in step.pinned]
        cut = set(spare[count - len(track_ids):]) if count < len(track_ids) else set()
        kept = [track_id for i, track_id in enumerate(track_ids) if i not in cut]
        candidate.track_ids = kept[:count]
        return f"Trimmed to {plural(count)}."
    # with no filters, top up from the artists already in the playlist.
    current = set(track_ids)
    groups, matches = selection.artist_groups, selection.matches
    if not selection.tests:
        groups = [{artist} for artist in dict.fromkeys(tracks[track_id]['artist'] for track_id in track_ids)]
        artists = set().union(*groups)
        matches = lambda t: t['artist'] in artists
    pool = [t for t in tracks.values() if t['id'] not in current and matches(t)]
    picked = pick(pool, groups, count - len(track_ids))
    if not picked:
        return "Found nothing more to add."
    candidate.track_ids = track_ids + [t['id'] for t in picked]
    return f"Added {plural(len(picked))} to make it {len(candidate.track_ids)}."

def sides(step: Step) -> List[Tuple[str, Test, bool]]:
    """The sides of an alternate: name, track test, and whether it is an album"""
    library, filters = step.library, step.filters
    found_sides = []
    for artist in filters.get('artists') or []:
        if found := library.find_artists(artist):
            found_sides.append((found[0], lambda t, found=set(found): t['artist'] in found, False))
    for album in filters.get('albums') or []:
        if found := library.find_albums(album):
            found_sides.append((min(found, key=len), lambda t, found=set(found): t['album'] in found, True))
    for genre in filters.get('genres') or []:
        for found in library.find_genres(genre):
            found_sides.append((found, lambda t, key=normalize(found): t['genre_key'] == key, False))
    return found_sides

def bring(step: Step, sides: List[Tuple[str, Test, bool]]) -> List[int]:
    """Tracks for the sides of an alternate that are not in the playlist yet"""
    current = set(step.candidate.track_ids)
    pools = [[t for t in step.tracks.values() if t['id'] not in current and on_side(t)] for _, on_side, _ in sides]
    albums = [
        sorted(pool, key=step.order or ORDER_KEYS['album'], reverse=step.descending)
        for pool, (_, _, whole) in zip(pools, sides) if whole
    ]
    # the other sides get as many tracks as the longest album.
    each = step.count // len(sides) if step.count else max(map(len, albums), default=PER_ARTIST_COUNT)
    brought = []
    for pool, (_, _, whole) in zip(pools, sides):
        brought += albums.pop(0) if whole else pick(pool, [], each)
    return list(dict.fromkeys(t['id'] for t in brought))

def alternate(step: Step) -> str:
    candidate, tracks, pinned = step.candidate, step.tracks, step.pinned
    track_ids, slots, brought = candidate.track_ids, step.slots, []
    if taking_part := sides(step):
        loose = [tracks[track_id] for track_id in track_ids if track_id not in pinned]
        if not step.positions and not any(on_side(t) for t in loose for _, on_side, _ in taking_part):
            brought = bring(step, taking_part)
            track_ids, slots = track_ids + brought, range(len(track_ids) + len(brought))
        names = [name for name, _, _ in taking_part] + ['the rest']
        turns: List[List[int]] = [[] for _ in names]
        for i in slots:
            track = tracks[track_ids[i]]
            if brought and track_ids[i] in pinned:
                continue
            side = next((n for n, (_, on_side, _) in enumerate(taking_part) if on_side(track)), len(taking_part))
            turns[side].append(i)
        # several named sides leave the rest where it is.
        if sum(map(bool, turns[:-1])) > 1:
            names, turns = names[:-1], turns[:-1]
    else:
        by_artist: Dict[str, List[int]] = {}
        for i in slots:
            by_artist.setdefault(tracks[track_ids[i]]['artist'], []).append(i)
        names, turns = list(by_artist), list(by_artist.values())
    names = [name for name, turn in zip(names, turns) if turn]
    turns = [turn for turn in turns if turn]
    if len(turns) < 2:
        return "Need at least two artists or albums to alternate."
    taking_turns = sorted(i for turn in turns for i in turn)
    waiting = [[track_ids[i] for i in turn] for turn in turns]
    alternated = []
    while any(waiting):
        alternated.extend(turn.pop(0) for turn in waiting if turn)
    summary = f"Alternated between {', '.join(names[:3])}{' and more' if len(names) > 3 else ''}."
    if brought:
        summary = f"Added {plural(len(brought))}. {summary}"
    candidate.track_ids = rearrange(track_ids, taking_turns, alternated)
    return summary

def sort(step: Step) -> str:
    candidate, chosen = step.candidate, step.chosen
    if not chosen:
        return "Found nothing to sort."
    by = step.call.get('by') if step.order else 'title'
    ordered = sorted(
        (candidate.track_ids[i] for i in chosen),
        key=lambda track_id: ORDER_KEYS[by](step.tracks[track_id]),
        reverse=step.descending,
    )
    candidate.track_ids = rearrange(candidate.track_ids, chosen, ordered)
    return f"Put {step.what(len(chosen))} in {'reverse ' if step.descending else ''}{by} order."

def reverse(step: Step) -> str:
    candidate, chosen = step.candidate, step.chosen
    if not chosen:
        return "Found nothing to reverse."
    candidate.track_ids = rearrange(candidate.track_ids, chosen, [candidate.track_ids[i] for i in reversed(chosen)])
    return f"Reversed {step.what(len(chosen))}."

def shuffle(step: Step) -> str:
    step.candidate.track_ids = random.sample(step.candidate.track_ids, len(step.candidate.track_ids))
    return "Shuffled."

def clear(step: Step) -> str:
    candidate = step.candidate
    # "start over with X" arrives as a clear that carries the filters for X.
    if step.selection.tests or step.count:
        previous, candidate.track_ids = candidate.track_ids, []
        summary = add(step, verb="Started over with" if previous else "Added")
        if not candidate.track_ids:
            candidate.track_ids = previous
        return summary
    candidate.track_ids = []
    return "Cleared the playlist."

def dedupe(step: Step) -> str:
    candidate, heard, kept = step.candidate, set(), []
    for track_id in candidate.track_ids:
        if (same := song(step.tracks[track_id])) not in heard:
            heard.add(same)
            kept.append(track_id)
    if not (repeats := len(candidate.track_ids) - len(kept)):
        return "Found no song twice."
    candidate.track_ids = kept
    return f"Removed {plural(repeats)} that repeated a song."

def cap(step: Step) -> str:
    candidate, per, most = step.candidate, step.per, step.count or 1
    taken, kept = Counter(), []
    for track_id in candidate.track_ids:
        group = GROUPS[per](step.tracks[track_id])
        taken[group] += 1
        if taken[group] <= most or track_id in step.pinned:
            kept.append(track_id)
    if not (removed := len(candidate.track_ids) - len(kept)):
        return f"No {per} had more than {plural(most)}."
    candidate.track_ids = kept
    return f"Kept at most {plural(most)} per {per}, removed {removed}."

def group(step: Step) -> str:
    candidate, per = step.candidate, step.per
    of = lambda track_id: GROUPS[per](step.tracks[track_id])
    # groups stay in the order their first tracks were in.
    first_seen = list(dict.fromkeys(map(of, candidate.track_ids)))
    candidate.track_ids = sorted(candidate.track_ids, key=lambda track_id: first_seen.index(of(track_id)))
    return f"Grouped the playlist by {per}."

def rename(step: Step) -> str:
    candidate, name = step.candidate, (step.call.get('name') or '').strip()
    if not name or (candidate.name and not NAMING.search(step.call.get('request') or '')):
        return ''
    previous, candidate.name = candidate.name, name
    # naming a new playlist is not news, the name shows up with the playlist.
    return f"Renamed the playlist to “{name}”." if previous else ''

def search(step: Step) -> str:
    found = [t for t in step.tracks.values() if step.selection.matches(t)]
    found.sort(key=step.order or ORDER_KEYS['artist'], reverse=step.descending)
    summary = ' '.join(filter(None, ["Found", plural(len(found)), step.selection.label, "in the library"]))
    shown = found[:min(step.count or SHOWN, SHOWN)]
    rows = [f"- {t['title']} - {t['artist']} ({t['album']}, {t['year'] or 'no year'})" for t in shown]
    if len(found) > len(shown):
        rows.append(f"... and {len(found) - len(shown)} more")
    step.reply = '\n'.join([f"{summary}:" if rows else f"{summary}.", *rows])
    return f"{summary}."

def browse(step: Step) -> str:
    field = step.call.get('field') if step.call.get('field') in FIELDS else 'artists'
    matching = [t for t in step.tracks.values() if step.selection.matches(t) and FIELDS[field](t)]
    counts = Counter(FIELDS[field](t) for t in matching)
    summary = ' '.join(filter(None, ["Found", str(len(counts)), field, step.selection.label, "in the library"]))
    rows = [f"- {value}, {plural(n)}" for value, n in counts.most_common(SHOWN)]
    if field == 'albums':
        released = {t['album']: t['year'] for t in matching if t['year']}
        rows = [f"- {album} ({released.get(album, 'no year')}), {plural(n)}" for album, n in counts.most_common(SHOWN)]
    if len(counts) > len(rows):
        rows.append(f"... and {len(counts) - len(rows)} more")
    step.reply = '\n'.join([f"{summary}:" if rows else f"{summary}.", *rows])
    return f"{summary}."

def unsupported(step: Step) -> str:
    said = (step.call.get('request') or '').strip()
    return f"Can't do this yet: “{said}”." if said else "Can't do this yet."

@dataclass
class Tool:
    """One thing an agent can do, or a stub of it when it has no `run` yet"""
    name: str
    description: str
    parameters: Tuple[str, ...] = ('request',)
    run: Optional[Callable[[Step], str]] = None
    required: Tuple[str, ...] = ('request',)

    def definition(self) -> Dict:
        """The tool as a function definition, the way chat models take them"""
        properties = {name: PARAMETERS[name] for name in self.parameters}
        parameters = {'type': 'object', 'properties': properties, 'required': list(self.required)}
        function = {'name': self.name, 'description': self.description, 'parameters': parameters}
        return {'type': 'function', 'function': function}

PARAMETERS = {
    'request': {'type': 'string'},
    'where': {'type': 'object'},
    'exclude': {'type': 'object'},
    'positions': {'type': 'array', 'items': {'type': 'integer'}},
    'count': {'type': 'integer'},
    'to': {'type': 'integer'},
    'by': {'type': 'string', 'enum': [order.name for order in ORDERS]},
    'descending': {'type': 'boolean'},
    'per': {'type': 'string', 'enum': list(GROUPS)},
    'minutes': {'type': 'number'},
    'name': {'type': 'string'},
    'field': {'type': 'string', 'enum': list(FIELDS)},
}

TOOLS = {tool.name: tool for tool in [
    Tool(
        'add',
        "Add tracks from the library that match `where`. With `by`, the first ones in that order instead of "
        "random ones. `to` or `positions` say where in the playlist they land.",
        ('request', 'where', 'exclude', 'count', 'by', 'descending', 'to', 'positions'), add,
    ),
    Tool(
        'remove',
        "Remove tracks from the playlist, picked by `where` or `positions`. With neither, removes the last "
        "`count` tracks.",
        ('request', 'where', 'exclude', 'positions', 'count'), remove,
    ),
    Tool(
        'keep',
        "Remove everything from the playlist except the tracks picked by `where` or `positions`.",
        ('request', 'where', 'exclude', 'positions'), keep,
    ),
    Tool(
        'move',
        "Move tracks that are already in the playlist to number `to`.",
        ('request', 'where', 'exclude', 'positions', 'to'), move,
    ),
    Tool('swap', "Swap the two tracks in `positions`.", ('request', 'positions'), swap),
    Tool(
        'resize',
        "Make the playlist exactly `count` tracks long, by trimming the end or adding more of the same artists.",
        ('request', 'count', 'where', 'exclude'), resize,
    ),
    Tool(
        'alternate',
        "Reorder the playlist so that its artists take turns. Name artists, albums or genres in `where` to make "
        "only those take turns, in that order, while every other track stays where it is. Any of them that are "
        "not in the playlist yet are added first.",
        ('request', 'where', 'positions', 'count', 'by', 'descending'), alternate,
    ),
    Tool(
        'sort',
        "Put the playlist in order of `by`. With `where` or `positions`, only those tracks.",
        ('request', 'by', 'descending', 'where', 'exclude', 'positions'), sort,
    ),
    Tool(
        'reverse',
        "Flip the order of the playlist. With `where` or `positions`, only those tracks.",
        ('request', 'where', 'exclude', 'positions'), reverse,
    ),
    Tool('shuffle', "Put the playlist in a random order.", run=shuffle),
    Tool(
        'clear',
        "Empty the playlist. With `where`, start the playlist over with the tracks that match.",
        ('request', 'where', 'exclude', 'count', 'by', 'descending'), clear,
    ),
    Tool(
        'dedupe',
        "Remove tracks that repeat a song the playlist already has, such as one from another release.",
        run=dedupe,
    ),
    Tool(
        'cap',
        "Allow at most `count` tracks per artist, album or genre, removing the later ones.",
        ('request', 'per', 'count'), cap,
    ),
    Tool(
        'group',
        "Bring the tracks of each artist, album or genre together, in the order they are already in.",
        ('request', 'per'), group,
    ),
    Tool(
        'spread',
        "Reorder the playlist so that tracks by the same artist, album or genre are kept apart.",
        ('request', 'per'),
    ),
    Tool(
        'fit',
        "Make the playlist run for about `minutes`, by trimming the end or adding more of the same artists.",
        ('request', 'minutes'),
    ),
    Tool(
        'rename',
        "Give the playlist a name. `request` is only for when the user asks for a name.",
        ('name', 'request'), rename, required=('name',),
    ),
    Tool(
        'search',
        "Look through the library without changing the playlist: lists the tracks that match `where`, in order "
        "of `by`.",
        ('where', 'exclude', 'by', 'descending', 'count'), search, required=(),
    ),
    Tool(
        'browse',
        "Look through the library without changing the playlist: lists the artists, albums, genres or years "
        "of the tracks that match `where`, with how many tracks each has.",
        ('field', 'where', 'exclude'), browse, required=('field',),
    ),
    Tool('unsupported', "The request asks for something that none of the other tools can do.", run=unsupported),
]}

def coerce(value, schema: Dict):
    """Fit an argument the model wrote to its schema, or None when it cannot be"""
    kind = schema.get('type')
    if value is None:
        return None
    try:
        if kind == 'array':
            items = value if isinstance(value, list) else [value]
            return [fitted for item in items if (fitted := coerce(item, schema['items'])) is not None]
        # the only objects are filters, which some models write as a string of JSON.
        if kind == 'object':
            filters = json.loads(value) if isinstance(value, str) else dict(value)
            fit = lambda name, given: coerce(given, PROPERTIES.get(name, {}))
            return {name: fitted for name, given in filters.items() if (fitted := fit(name, given)) is not None}
        if kind == 'integer':
            return int(float(value))
        if kind == 'number':
            return float(value)
        if kind == 'boolean':
            return value if isinstance(value, bool) else str(value).lower() == 'true'
        if kind == 'string':
            return str(value)
    except (TypeError, ValueError):
        return None
    return value

class Toolbox:
    """The tools an agent has for editing a playlist from a library"""

    def __init__(self, library: Library):
        self.library = library

    def definitions(self) -> List[Dict]:
        return [tool.definition() for tool in TOOLS.values()]

    def repair(self, call: Dict) -> Dict:
        """Fit a call to its tool, and drop what the quoted request never asked for"""
        if (tool := TOOLS.get(call.get('action'))) is None:
            return {'action': call.get('action'), 'request': coerce(call.get('request'), PARAMETERS['request']) or ''}
        repaired = {'action': tool.name}
        for name in tool.parameters:
            if (fitted := coerce(call.get(name), PARAMETERS[name])) is not None:
                repaired[name] = fitted
        # a filter written beside `where` belongs in it.
        if 'where' in tool.parameters:
            beside = coerce({name: call[name] for name in PROPERTIES if name in call}, PARAMETERS['where'])
            repaired['where'] = {**repaired.get('where', {}), **beside}

        # a name filed as the wrong kind moves to the kind the library has it as.
        where = repaired.get('where', {})
        find = {'artists': self.library.find_artists, 'albums': self.library.find_albums}
        for kind, other in (('artists', 'albums'), ('albums', 'artists')):
            for name in list(where.get(kind) or []):
                if not find[kind](name) and find[other](name):
                    where[kind] = [given for given in where[kind] if given != name]
                    where[other] = [*(where.get(other) or []), name]

        if said := repaired.get('request'):
            if not NEGATIVES.search(said):
                repaired.pop('exclude', None)
            if tool.name == 'add' and not NUMBERS.search(said):
                repaired.pop('positions', None)
            if tool.name == 'alternate' and where.get('genres'):
                where['genres'] = [genre for genre in where['genres'] if normalize(genre) in normalize(said)]
        for filters in ('where', 'exclude'):
            if filters in repaired:
                repaired[filters] = {name: value for name, value in repaired[filters].items() if value != []}
        return {name: value for name, value in repaired.items() if value not in ([], {})}

    def apply(self, call: Dict, candidate: Candidate) -> Dict:
        """Carry out one call, hold pinned tracks in place, and describe the result"""
        call = self.repair(call)
        tool = TOOLS.get(call.get('action'))
        selection = select(self.library, call.get('where') or {}, call.get('exclude') or {})
        step = Step(self.library, candidate, call, selection)
        before = list(candidate.track_ids)

        gaps = selection.gaps
        if tool is None:
            gaps.append(f"There is no tool called {call.get('action')}.")
        elif tool.run is None:
            gaps.append(f"The {tool.name} tool is not built yet.")
        if (by := call.get('by')) and by not in ORDER_KEYS:
            gaps.append(f"Can't sort by {by} yet.")
        if gaps:
            selection.notes += gaps
            summary = unsupported(step)
        else:
            summary = tool.run(step)

        candidate.pins.update(placed(call, before, candidate.track_ids))
        candidate.track_ids = hold(candidate.track_ids, candidate.pins)
        return {
            'operation': call,
            'track_ids': candidate.track_ids,
            'summary': summary,
            'notes': selection.notes,
            'suggestions': selection.suggestions,
            'reply': step.reply or ' '.join(filter(None, [summary, *selection.notes])) or "Done.",
        }
