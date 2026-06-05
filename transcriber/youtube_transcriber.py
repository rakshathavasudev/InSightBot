import os
import ssl
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi

_yt_api = YouTubeTranscriptApi()


class YouTubeTranscriber:
    def __init__(self, url, transcript_language='en', output_dir='resources/transcripts'):
        self.url = url
        self.transcript_language = transcript_language
        self.output_dir = output_dir

        if 'playlist?list=' in url:
            self.is_playlist = True
            self.playlist = []
            ssl._create_default_https_context = ssl._create_unverified_context
            for u in Playlist(url):
                self.playlist.append(u)
        elif 'watch?v=' in url or 'youtu.be/' in url:
            self.is_playlist = False
            if 'watch?v=' in url:
                self.video_id = url.split('watch?v=')[1].split('&')[0]
            else:
                self.video_id = url.split('youtu.be/')[1].split('?')[0]
        else:
            raise ValueError("Invalid URL. Provide a valid YouTube playlist or video URL.")

        os.makedirs(self.output_dir, exist_ok=True)
        if self.is_playlist:
            print(f"Found {len(self.playlist)} videos in the playlist.")

    def fetch_transcript(self, video_id):
        try:
            transcript_list = _yt_api.list(video_id)
            print(f"Available transcripts for video {video_id}:")
            for t in transcript_list:
                print(f"  - {t.language} ({t.language_code}) - Generated: {t.is_generated}")

            fetched = None
            try:
                fetched = transcript_list.find_transcript([self.transcript_language]).fetch()
                print(f"Fetched {self.transcript_language} transcript for {video_id}")
            except Exception as e1:
                print(f"Preferred language failed ({e1}), trying fallback...")
                for t in transcript_list:
                    try:
                        fetched = t.fetch()
                        print(f"Fetched {t.language_code} transcript for {video_id} (fallback)")
                        break
                    except Exception as e2:
                        print(f"  {t.language_code} failed: {e2}")

            if fetched is None:
                print(f"No transcripts available for video ID {video_id}")
                return None

            snippets = [{'start': s.start, 'text': s.text} for s in fetched]
            print(f"Successfully fetched {len(snippets)} snippets for video ID {video_id}")
            return snippets
        except Exception as e:
            print(f"Could not retrieve transcript for video ID {video_id}: {e}")
            return None

    def save_transcript_to_file(self, video_id, transcript):
        file_path = os.path.join(self.output_dir, f"{video_id}_transcript.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for line in transcript:
                f.write(f"{line['start']}: {line['text']}\n")
        print(f"Transcript saved for video ID: {video_id}")

    def transcribe_playlist(self):
        success_count = 0
        for video_url in self.playlist:
            if 'watch?v=' in video_url:
                video_id = video_url.split('watch?v=')[1].split('&')[0]
            elif 'youtu.be/' in video_url:
                video_id = video_url.split('youtu.be/')[1].split('?')[0]
            else:
                print(f"Skipping unrecognised URL: {video_url}")
                continue
            transcript = self.fetch_transcript(video_id)
            if transcript:
                self.save_transcript_to_file(video_id, transcript)
                success_count += 1
            else:
                print(f"No transcript file created for video ID {video_id}")
        print(f"Successfully transcribed {success_count} out of {len(self.playlist)} videos")
        return success_count > 0

    def transcribe_single_video(self):
        transcript = self.fetch_transcript(self.video_id)
        if transcript:
            self.save_transcript_to_file(self.video_id, transcript)
            return True
        print(f"No transcript file created for video ID {self.video_id}")
        return False

    def transcribe(self):
        if self.is_playlist:
            return self.transcribe_playlist()
        return self.transcribe_single_video()
