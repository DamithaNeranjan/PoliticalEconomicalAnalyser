import json
import os
import tempfile
from datetime import datetime, timedelta

import whisper
import yt_dlp
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langdetect import detect
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables from .env file
load_dotenv()

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Search YouTube (last 3 months)
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

THREE_MONTHS_AGO = (datetime.now() - timedelta(days=1)).isoformat("T") + "Z"

QUERY = (
    "Sri Lanka politics OR economy OR parliament OR election OR budget "
    "OR IMF OR inflation OR government"
)


def search_videos():
    request = youtube.search().list(
        q=QUERY,
        part="snippet",
        type="video",
        eventType="completed",
        regionCode="LK",
        publishedAfter=THREE_MONTHS_AGO,
        maxResults=1,
        relevanceLanguage="si",
        videoDuration="short"
    )
    return request.execute()["items"]


# Caption extraction (Sinhala + English)
def get_captions(video_id):
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        print("transcripts")

        if transcripts.find_manually_created_transcript(['si']):
            return transcripts.find_manually_created_transcript(['si']).fetch()

        if transcripts.find_manually_created_transcript(['en']):
            return transcripts.find_manually_created_transcript(['en']).fetch()

        return transcripts.find_generated_transcript(['si', 'en']).fetch()

    except Exception:
        print("No captions found")
        return None


# Audio transcription fallback (Whisper)
model = whisper.load_model("base")


def transcribe_audio(video_url):
    video_id = video_url.split("v=")[-1]
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_path.replace(".wav", ".%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,

        # JS runtime
        "js_runtimes": {
            "node": {}
        },

        # Avoid broken SABR clients
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        },

        # Reduce throttling
        "concurrent_fragment_downloads": 1,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not created: {audio_path}")

    result = model.transcribe(
        audio_path,
        task="translate",  # Sinhala → English directly
        fp16=False  # CPU-safe
    )

    return result["text"]


# Language detection + Sinhala → English translation
def normalize_text(text):
    lang = detect(text)

    if lang == "si":
        return GoogleTranslator(source="si", target="en").translate(text)

    return text


# Main pipeline
def run_pipeline():
    results = []

    videos = search_videos()

    for v in tqdm(videos):
        video_id = v["id"]["videoId"]
        title = v["snippet"]["title"]
        description = v["snippet"]["description"]
        url = f"https://www.youtube.com/watch?v={video_id}"

        captions = get_captions(video_id)

        if captions:
            text = " ".join([c["text"] for c in captions])
        else:
            text = transcribe_audio(url)
            print(f"⚠ No captions for video {url}, skipping transcription")
            # continue  # Skip if no captions and no transcription

        text_en = normalize_text(text)

        results.append({
            "video_id": video_id,
            "title": title,
            "published": v["snippet"]["publishedAt"],
            "source": "YouTube",
            "url": url,
            "content_en": text_en
        })

    with open("sri_lanka_political_economic_text.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✔ Data extraction complete")


# Run the pipeline
run_pipeline()
