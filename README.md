# PheltPhished

Voice phishing demo bot for security awareness training (Agent Olympics).

The bot joins a Google Meet call and uses AI to simulate a social engineering attack.

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (required by pydub for audio processing)
- [ngrok](https://ngrok.com/) account (free tier works)
- [Recall.ai](https://recall.ai/) API key
- [OpenAI](https://platform.openai.com/) API key with Realtime API access

### Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

### Install ngrok

```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/download
```

## Setup

1. **Clone and set up virtual environment**

   ```bash
   git clone <repo-url>
   cd PheltPhished
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Create `.env` file**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` with your keys:

   ```
   RECALL_API_KEY=your_recall_api_key
   OPENAI_API_KEY=your_openai_api_key
   NGROK_URL=wss://xxxx.ngrok.io
   ```

3. **Start ngrok** (in a separate terminal)

   ```bash
   ngrok http 8000
   ```

   Copy the `wss://` URL (convert `https://` to `wss://`) and paste it into your `.env` as `NGROK_URL`.

## Running the Bot

```bash
python main.py 'https://meet.google.com/xxx-yyyy-zzz'
```

The bot will:
1. Start a WebSocket server on port 8000
2. Create a Recall.ai bot that joins the meeting
3. Stream audio between the meeting and OpenAI's Realtime API
4. Print transcripts to the console

## How It Works

```
Google Meet → Recall.ai → Your Server (port 8000) → OpenAI Realtime API
                 ↑                                          ↓
                 └──────────── AI voice response ───────────┘
```

1. Recall.ai bot joins the meeting and streams audio to your server via WebSocket
2. Your server resamples audio (16kHz → 24kHz) and forwards to OpenAI
3. OpenAI generates voice responses using GPT-4o
4. Your server converts audio (PCM → MP3) and sends back to Recall.ai
5. Recall.ai plays the audio in the meeting

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pydub` errors | Make sure ffmpeg is installed and in your PATH |
| Bot doesn't join | Check your Recall.ai API key and meeting URL format |
| No audio flowing | Verify ngrok is running and NGROK_URL uses `wss://` (not `https://`) |
| OpenAI errors | Ensure your API key has Realtime API access |
