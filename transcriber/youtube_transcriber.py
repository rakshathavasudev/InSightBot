import os
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi

class YouTubeTranscriber:
    def __init__(self, url, transcript_language='en', output_dir='transcripts'):
        """
        Initializes the YouTubeTranscriber with a URL (playlist or single video), transcript language, and output directory.

        :param url: URL of the YouTube playlist or single video
        :param transcript_language: Preferred language for transcripts (default is 'en' for English)
        :param output_dir: Directory where transcripts will be saved
        """
        self.url = url
        self.transcript_language = transcript_language
        self.output_dir = output_dir

        # Check if the URL is for a playlist or a single video
        if 'playlist?list=' in url:
            self.is_playlist = True
            self.playlist = Playlist(url)
            print('self.playlist', self.playlist)
        elif 'watch?v=' or 'youtu.be/' in url:
            self.is_playlist = False
            self.video_id = url.split('=')[-1]
        else:
            raise ValueError("Invalid URL. Provide a valid YouTube playlist or video URL.")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        if self.is_playlist:
            print(f"Found {len(self.playlist.video_urls)} videos in the playlist.")

    def fetch_transcript(self, video_id):
        """
        Fetches the transcript for a single video in the specified language.

        :param video_id: YouTube video ID
        :return: Transcript as a list of dictionaries with 'start' and 'text' keys
        """
        try:
            # Fetch transcript with the specified language
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[self.transcript_language])
            return transcript
        except Exception as e:
            print(f"Could not retrieve transcript for video ID {video_id}: {e}")
            return None

    def save_transcript_to_file(self, video_id, transcript):
        """
        Saves the transcript of a video to a text file.

        :param video_id: YouTube video ID
        :param transcript: Transcript data to save
        """
        file_path = os.path.join(self.output_dir, f"{video_id}_transcript.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            for line in transcript:
                file.write(f"{line['start']}: {line['text']}\n")
        print(f"Transcript saved for video ID: {video_id}")

    def transcribe_playlist(self):
        """
        Processes each video in the playlist to fetch and save transcripts.
        """
        for video_url in self.playlist.video_urls:
            # Extract video ID from URL
            video_id = video_url.split('=')[-1]
            # Fetch and save the transcript
            transcript = self.fetch_transcript(video_id)
            if transcript:
                self.save_transcript_to_file(video_id, transcript)

    def transcribe_single_video(self):
        """
        Fetches and saves the transcript for a single YouTube video.
        """
        # Fetch and save the transcript
        transcript = self.fetch_transcript(self.video_id)
        if transcript:
            self.save_transcript_to_file(self.video_id, transcript)

    def transcribe(self):
        """
        Determines if the URL is a playlist or single video and processes accordingly.
        """
        if self.is_playlist:
            self.transcribe_playlist()
        else:
            self.transcribe_single_video()
