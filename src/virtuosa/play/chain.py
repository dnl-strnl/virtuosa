import json
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
import tqdm
from typing import Dict, List

class MusicLibraryChain:
    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://127.0.0.1:11434"
    ):

        self.llm = Ollama(model=model_name, base_url=base_url)
        self.embed_model = OllamaEmbeddings(
            model=model_name,
            base_url=base_url,
            show_progress=True,
        )

        self.vector_store = None
        self.retrieval_chain = None
        self.k = 50

    def prepare_documents(self, music_data: List[Dict]) -> List[Document]:
        """Convert music library data into LangChain documents with audio metadata"""

        documents = []
        for track in tqdm.tqdm(music_data):
            # create a track description with all available metadata.
            content = (
                f"Track: {track.get('title', 'Unknown')} "
                f"by {track.get('artist', 'Unknown')} "
                f"from album {track.get('album', 'Unknown')}. "
                f"Album Artist: {track.get('album_artist', 'Unknown')}. "
                f"Composer: {track.get('composer', 'Unknown')}. "
                f"Genre: {track.get('genre', 'Unknown')}. "
                f"Year: {track.get('year', 'Unknown')}. "
                f"Duration: {track.get('duration', 0)} seconds. "
                f"BPM: {track.get('bpm', 'Unknown')}. "
            )

            # store complete track data in metadata for retrieval.
            metadata = {
                'filepath':  track.get('filepath', None),
                'title': track.get('title', 'Unknown'),
                'artist': track.get('artist', 'Unknown'),
                'album': track.get('album', 'Unknown'),
                'album_artist': track.get('album_artist', 'Unknown'),
                'composer': track.get('composer', 'Unknown'),
                'genre': track.get('genre', 'Unknown'),
                'year': track.get('year', 'Unknown'),
                'track_number': track.get('track_number', '0'),
                'disc_number': track.get('disc_number', '0'),
                'bpm': track.get('bpm', 'Unknown'),
                'duration': track.get('duration', 0),
                'bitrate': track.get('bitrate', 0)
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def initialize_chain(self, music_data: List[Dict]) -> None:
        """Initialize the retrieval chain with music library data"""

        documents = self.prepare_documents(music_data)

        self.vector_store = Chroma.from_documents(
            documents,
            self.embed_model
        )

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )

        combine_docs_chain = create_stuff_documents_chain(
            self.llm,
            retrieval_qa_chat_prompt = hub.pull(
                "langchain-ai/retrieval-qa-chat"
            )
        )

        self.retrieval_chain = create_retrieval_chain(
            retriever,
            combine_docs_chain
        )

    def generate_playlist(self, description: str) -> Dict:
        """Generate a playlist based on voice description"""
        if not self.retrieval_chain:
            raise ValueError("Chain not initialized. Call `initialize_chain` first.")

        prompt = f"""
        Based on the following request: "{description}"
        Create a playlist using only the available songs in the library.
        Return a JSON response with the following structure:
        {{
            "tracks": [
                {{"filepath": "relative/path/to/song.mp3",
                  "title": "Song Title",
                  "artist": "Artist Name"}}
            ]
        }}
        Include only songs that match the request's mood and style.
        """

        response = self.retrieval_chain.invoke(dict(input=prompt))

        # parse response and extract suggested tracks.
        try:
            answer_text = response['answer']

            start_idx = answer_text.find('{')
            end_idx = answer_text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No valid JSON found in response.")

            json_str = answer_text[start_idx:end_idx]
            playlist_data = json.loads(json_str)

            return {
                'name': f"{description[:30]}...",
                'tracks': playlist_data['tracks'],
                'description': description,
            }

        except Exception as playlist_parse_error:
            raise ValueError(f"{playlist_parse_error=}")
