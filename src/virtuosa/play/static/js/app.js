class MP3Player {
    constructor() {
        this.audioPlayer = document.getElementById('audio-player');
        this.playlist = null;    // the saved playlist being edited, or null for a new one
        this.tracks = [];        // what is on screen, which leads `playlist` mid-command
        this.rows = new Map();   // each track's <li>, by track id
        this.playingId = null;
        this.busy = false;
        this.recorder = null;

        this.initializeElements();
        this.setupEventListeners();
        this.renderPlaylist();
        this.loadPlaylists();
    }

    initializeElements() {
        this.voicePlaylistBtn = document.getElementById('voice-playlist-btn');
        this.recordingStatus = document.getElementById('recording-status');
        this.commandForm = document.getElementById('command-form');
        this.commandInput = document.getElementById('command-input');
        this.activity = document.getElementById('activity');

        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.nowPlaying = document.getElementById('now-playing');

        this.newPlaylistBtn = document.getElementById('new-playlist-btn');
        this.deletePlaylistBtn = document.getElementById('delete-playlist-btn');
        this.playlistList = document.getElementById('playlist-list');
        this.playlistName = document.getElementById('playlist-name');
        this.playlistMeta = document.getElementById('playlist-meta');
        this.trackList = document.getElementById('track-list');
    }

    setupEventListeners() {
        this.voicePlaylistBtn.addEventListener('click', () => this.toggleRecording());
        this.commandForm.addEventListener('submit', (event) => {
            event.preventDefault();
            const text = this.commandInput.value.trim();
            if (!text || this.busy) return;

            const formData = new FormData();
            formData.append('text', text);
            this.commandInput.value = '';
            this.sendCommand(formData);
        });

        this.prevBtn.addEventListener('click', () => this.step(-1));
        this.nextBtn.addEventListener('click', () => this.step(1));
        this.audioPlayer.addEventListener('ended', () => this.step(1));

        this.newPlaylistBtn.addEventListener('click', () => {
            this.activity.replaceChildren();
            this.openPlaylist(null);
        });
        this.deletePlaylistBtn.addEventListener('click', () => this.deletePlaylist());
    }

    async toggleRecording() {
        if (this.recorder && this.recorder.state === 'recording') {
            this.recorder.stop();
            return;
        }
        if (this.busy) return;

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (error) {
            console.error('Error setting up voice recording:', error);
            this.recordingStatus.textContent = 'Failed to access microphone';
            return;
        }

        const audioChunks = [];
        this.recorder = new MediaRecorder(stream);
        this.recorder.ondataavailable = (event) => audioChunks.push(event.data);
        this.recorder.onstop = () => {
            stream.getTracks().forEach(track => track.stop());
            this.voicePlaylistBtn.classList.remove('recording');

            const formData = new FormData();
            formData.append('audio', new Blob(audioChunks, { type: this.recorder.mimeType }));
            this.sendCommand(formData, 'Transcribing...');
        };

        this.recorder.start();
        this.voicePlaylistBtn.classList.add('recording');
        this.recordingStatus.textContent = this.playlist
            ? `Recording an edit to “${this.playlist.name}”... Click again to stop`
            : 'Recording... Click again to stop';
    }

    async sendCommand(formData, status = 'Thinking...') {
        if (this.busy) return;
        this.setBusy(true);
        this.recordingStatus.textContent = status;
        this.activity.replaceChildren();
        if (this.playlist) formData.append('playlist_id', this.playlist.id);

        try {
            const response = await fetch('/api/command', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Failed to send command');

            // One JSON event per line, as each step happens
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                lines.filter(Boolean).forEach(line => this.handleEvent(JSON.parse(line)));
            }
        } catch (error) {
            console.error('Error sending command:', error);
            this.handleEvent({ event: 'error', error: error.message });
        }

        this.setBusy(false);
    }

    handleEvent(event) {
        if (event.event === 'heard') {
            this.log(`“${event.text}”`, 'heard');
            this.recordingStatus.textContent = 'Thinking...';
        } else if (event.event === 'op') {
            this.logResult(event);
            this.renderTracks(event.tracks);
        } else if (event.event === 'done') {
            this.playlist = event.playlist;
            this.renderPlaylist();
            this.loadPlaylists();
        } else if (event.event === 'error') {
            this.log(event.error, 'error');
            // Nothing was saved, so go back to what the server has
            this.openPlaylist(this.playlist ? this.playlist.id : null);
        }
    }

    log(text, kind = '') {
        const item = document.createElement('li');
        item.textContent = text;
        item.className = kind;
        this.activity.appendChild(item);
        return item;
    }

    // Logs what an operation did, with buttons for names that were not found
    logResult(result) {
        this.log(result.summary);
        result.notes.forEach(note => this.log(note, 'note'));
        result.suggestions.forEach(suggestion => {
            const item = this.log(`Did you mean, instead of “${suggestion.heard}”:`, 'suggestion');
            suggestion.options.forEach(option => {
                const button = document.createElement('button');
                button.textContent = option;
                button.addEventListener('click', () => this.correct(result.operation, suggestion, option, item));
                item.appendChild(button);
            });
        });
    }

    // Redo an operation with the name picked from its suggestions
    async correct(operation, suggestion, choice, item) {
        if (this.busy) return;
        this.setBusy(true);
        try {
            const response = await fetch('/api/correct', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    playlist_id: this.playlist ? this.playlist.id : null,
                    operation: operation,
                    field: suggestion.field,
                    heard: suggestion.heard,
                    choice: choice,
                }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error);

            item.remove();
            this.log(`“${suggestion.heard}” is ${choice}, got it.`, 'heard');
            this.logResult(result);
            this.playlist = result.playlist;
            this.renderPlaylist();
            this.loadPlaylists();
        } catch (error) {
            console.error('Error correcting:', error);
            this.log(error.message, 'error');
        }
        this.setBusy(false);
    }

    setBusy(busy) {
        this.busy = busy;
        this.commandInput.disabled = busy;
        this.voicePlaylistBtn.disabled = busy;
        if (!busy) {
            this.showMode();
            this.commandInput.focus();
        }
    }

    // The microphone starts a new playlist, the pencil edits the open one
    showMode() {
        const playlist = this.playlist;
        this.voicePlaylistBtn.classList.toggle('editing', Boolean(playlist));
        this.voicePlaylistBtn.title = playlist ? 'Edit this playlist by voice' : 'Describe a playlist by voice';
        this.recordingStatus.textContent = playlist
            ? `Editing “${playlist.name}”. Click the pencil and say what to change, or use + New playlist to start another.`
            : 'Click the microphone and say what you want to hear.';
        this.commandInput.placeholder = playlist
            ? 'Edit it: “swap 2 and 5”, “move the last song to the top”, “remove track 3”, “make it 20 tracks”'
            : 'Describe a playlist: name artists, albums or a mood, like “something chill from the nineties”';
    }

    async loadPlaylists() {
        try {
            const response = await fetch('/api/playlists');
            this.renderPlaylists(await response.json());
        } catch (error) {
            console.error('Error loading playlists:', error);
        }
    }

    async openPlaylist(id) {
        if (id === null) {
            this.playlist = null;
        } else {
            try {
                const response = await fetch(`/api/playlists/${id}`);
                if (!response.ok) throw new Error('Playlist not found');
                this.playlist = await response.json();
            } catch (error) {
                console.error('Error loading playlist:', error);
                this.playlist = null;
            }
        }
        this.renderPlaylist();
        this.loadPlaylists();
    }

    async deletePlaylist() {
        if (!this.playlist || !confirm(`Delete “${this.playlist.name}”?`)) return;
        await fetch(`/api/playlists/${this.playlist.id}`, { method: 'DELETE' });
        this.activity.replaceChildren();
        this.openPlaylist(null);
    }

    renderPlaylists(playlists) {
        this.playlistList.replaceChildren(...playlists.map(playlist => {
            const item = document.createElement('li');
            item.className = 'playlist-item';
            item.classList.toggle('active', Boolean(this.playlist) && this.playlist.id === playlist.id);
            item.addEventListener('click', () => {
                this.activity.replaceChildren();
                this.openPlaylist(playlist.id);
            });

            const name = document.createElement('span');
            name.textContent = playlist.name;
            const count = document.createElement('span');
            count.textContent = playlist.count;
            item.append(name, count);
            return item;
        }));
    }

    renderPlaylist() {
        const playlist = this.playlist;
        this.playlistName.textContent = playlist ? playlist.name : 'New playlist';
        this.deletePlaylistBtn.hidden = !playlist;
        if (!this.busy) this.showMode();
        this.renderTracks(playlist ? playlist.tracks : []);
    }

    // New rows fade in, missing rows fade out, moved rows slide into place
    renderTracks(tracks) {
        const before = new Map([...this.rows].map(([id, row]) => [id, row.getBoundingClientRect().top]));
        const wanted = new Set(tracks.map(track => track.id));
        for (const [id, row] of this.rows) {
            if (wanted.has(id)) continue;
            this.rows.delete(id);
            row.style.height = `${row.offsetHeight}px`;
            row.classList.add('leaving');
            setTimeout(() => row.remove(), 300);
        }

        let cursor = this.trackList.firstElementChild;
        let fresh = 0;
        for (const track of tracks) {
            while (cursor && cursor.classList.contains('leaving')) cursor = cursor.nextElementSibling;

            let row = this.rows.get(track.id);
            if (!row) {
                row = this.createRow(track);
                row.style.animationDelay = `${Math.min(fresh++ * 40, 600)}ms`;
                this.rows.set(track.id, row);
            }
            if (row === cursor) {
                cursor = cursor.nextElementSibling;
            } else {
                this.trackList.insertBefore(row, cursor);
            }
        }

        for (const [id, top] of before) {
            const row = this.rows.get(id);
            const moved = row ? top - row.getBoundingClientRect().top : 0;
            if (moved) {
                row.animate(
                    [{ transform: `translateY(${moved}px)` }, { transform: 'none' }],
                    { duration: 350, easing: 'ease-out' }
                );
            }
        }

        this.tracks = tracks;
        const seconds = tracks.reduce((total, track) => total + track.duration, 0);
        this.playlistMeta.textContent = tracks.length
            ? `${tracks.length} ${tracks.length === 1 ? 'track' : 'tracks'} · ${Math.round(seconds / 60)} min`
            : 'No tracks yet.';
    }

    createRow(track) {
        const row = document.createElement('li');
        row.className = 'track';
        row.classList.toggle('playing', track.id === this.playingId);
        row.addEventListener('click', () => this.playTrack(track.id));

        const cells = {
            title: track.title,
            artist: track.artist,
            album: track.album,
            duration: this.formatTime(track.duration),
        };
        for (const [name, text] of Object.entries(cells)) {
            const cell = document.createElement('span');
            cell.className = name;
            cell.textContent = text;
            row.appendChild(cell);
        }
        return row;
    }

    playTrack(id) {
        const track = this.tracks.find(track => track.id === id);
        if (!track) return;

        if (this.rows.has(this.playingId)) this.rows.get(this.playingId).classList.remove('playing');
        this.playingId = id;
        this.rows.get(id).classList.add('playing');

        this.nowPlaying.textContent = `${track.title} · ${track.artist}`;
        this.audioPlayer.src = `/stream/${id}`;
        this.audioPlayer.play();
    }

    step(direction) {
        const index = this.tracks.findIndex(track => track.id === this.playingId);
        const next = this.tracks[index + direction];
        if (next) this.playTrack(next.id);
    }

    formatTime(seconds) {
        return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    }
}

const player = new MP3Player();
