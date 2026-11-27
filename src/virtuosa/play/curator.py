import json
import logging
import ollama
from pathlib import Path
from typing import Dict, Iterator, List

from virtuosa.play.filters import FILTERS, ORDERS, guide
from virtuosa.play.library import Library
from virtuosa.play.tools import Candidate, Toolbox, plural


log = logging.getLogger(__name__)

SYSTEM_PROMPT = Path(__file__).with_name('system_prompt.txt').read_text(encoding='utf-8')

MAX_LISTED = 100
# room for the prompt, the tools, MAX_LISTED tracks and the answer.
CONTEXT_TOKENS = 8192
# turns the model gets to call tools and read what they did.
MAX_ROUNDS = 4

class PlaylistCurator:
    def __init__(
        self,
        library: Library,
        model_name: str = "qwen3:8b",
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.library = library
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        self.toolbox = Toolbox(library)
        self.system_prompt = SYSTEM_PROMPT.replace('{filters}', guide(FILTERS)).replace('{orders}', guide(ORDERS))

    def warm_up(self) -> None:
        """Load the model into memory so the first request does not wait for it"""
        try:
            self.client.generate(model=self.model_name, options={'num_ctx': CONTEXT_TOKENS}, keep_alive='30m')
        except Exception as warm_up_error:
            log.warning(f"{warm_up_error=}")

    def describe(self, name: str, track_ids: List[int]) -> str:
        """The playlist as a numbered list, so the model can refer to slots"""
        if not track_ids:
            return f'Playlist "{name}" is empty.' if name else "Playlist: new, empty and not named yet."
        tracks = [self.library.tracks[track_id] for track_id in track_ids]
        listing = [f"{i}. {t['title']} - {t['artist']}" for i, t in enumerate(tracks[:MAX_LISTED], start=1)]
        if len(tracks) > MAX_LISTED:
            listing.append(f"... and {len(tracks) - MAX_LISTED} more, up to number {len(tracks)}")
        return '\n'.join([f'Playlist "{name}" has {plural(len(tracks))}:', *listing])

    def edit(self, request: str, name: str, track_ids: List[int]) -> Iterator[Dict]:
        """Yield an `op` event per tool call as the model makes it, then a `plan`"""
        candidate = Candidate(name, [track_id for track_id in track_ids if track_id in self.library.tracks])
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': (
                f"{self.describe(candidate.name, candidate.track_ids)}\n"
                f"Genres in the library: {', '.join(self.library.genres)}\n"
                f"Similar library artists: {', '.join(self.library.sounds_like(request)) or 'none'}\n"
                f"Request: {request}"
            )},
        ]

        repeated, carried_out, earlier = None, set(), set()
        for _ in range(MAX_ROUNDS):
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                tools=self.toolbox.definitions(),
                # a small context stays on the GPU, and the cap stops a runaway answer.
                options={'temperature': 0, 'num_ctx': CONTEXT_TOKENS, 'num_predict': 600},
                think=False,
                keep_alive='30m',
                stream=True,
            )

            made, replies, carried = [], [], 0
            for part in response:
                for tool_call in part.message.tool_calls or []:
                    call = {'action': tool_call.function.name, **tool_call.function.arguments}
                    log.info(f"call: {call}")
                    made.append(call)
                    # the model sometimes makes the same call twice in a row, or again in a later round.
                    asked = {**call, 'request': ''}
                    if asked == repeated or json.dumps(call, sort_keys=True, default=str) in earlier:
                        replies.append("Already done.")
                        continue
                    repeated = asked
                    # or repeats words it has already carried out.
                    said = str(call.get('request') or '').strip().lower()
                    again = any(len(done) > 10 and done in said for done in carried_out)
                    if again and (call['action'] == 'unsupported' or said == request.strip().lower()):
                        replies.append("Already done.")
                        continue
                    carried_out.add(said)
                    result = self.toolbox.apply(call, candidate)
                    carried += 1
                    replies.append(result.pop('reply'))
                    if result['summary']:
                        yield {'event': 'op', **result}

            # a round of nothing new means the model is done.
            if not carried:
                break
            earlier.update(json.dumps(call, sort_keys=True, default=str) for call in made)
            tool_calls = [
                {'function': {'name': call['action'], 'arguments': {k: v for k, v in call.items() if k != 'action'}}}
                for call in made
            ]
            messages.append({'role': 'assistant', 'tool_calls': tool_calls})
            messages += [
                {'role': 'tool', 'tool_name': call['action'], 'content': reply} for call, reply in zip(made, replies)
            ]

        yield {'event': 'plan', 'name': candidate.name or f"{request[:30]}...", 'track_ids': candidate.track_ids}

    def apply(self, operation: Dict, track_ids: List[int]) -> Dict:
        """Carry out one call on a saved playlist, outside of a request"""
        result = self.toolbox.apply(operation, Candidate('', list(track_ids)))
        del result['reply']
        return result
