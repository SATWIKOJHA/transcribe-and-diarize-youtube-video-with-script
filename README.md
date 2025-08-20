You Can Directly go and Watch the Dataset with the given link:-
https://huggingface.co/datasets/ojhasatwik/bhojpuri_commentry_ipl

YouTube Video to Text with Speaker Timestamps 🎙️➡️📝
This project is a powerful Python script that takes any public YouTube video, automatically downloads the audio, identifies who is speaking and when, and converts their speech into a detailed text transcript.

It's perfect for creating meeting notes, interview transcripts, or content from podcasts and educational videos.

What it Does
Downloads Audio: Grabs the audio from any YouTube URL you provide.

Identifies Speakers: Uses a sophisticated AI model (pyannote.audio) to detect different speakers in the audio.

Transcribes Speech: Converts the spoken words from each speaker into text.

Creates Detailed Output: Generates a neat .csv file for each video, containing the start time, end time, speaker ID, and the transcribed text.

⚙️ How It Works (The Important Part!)
This script is designed to be efficient by splitting up the hard work. It connects to two main services:

Hugging Face: To download the speaker identification model (pyannote/speaker-diarization-3.1). You'll need a free account to get an access token.

A Whisper API: For the actual speech-to-text conversion.

A Quick Note on the Whisper API
This script does not run the Whisper model by itself. Instead, it sends the audio segments to a separate Whisper API service that you run on your own.

Why? This makes the main script much lighter and faster. You can have a powerful machine (with a good GPU) dedicated to running the Whisper model, and this script can run on any computer.

What you need to do:
You need to set up your own Whisper API endpoint. A great way to do this is by using a pre-built solution like the one from OpenAI's Whisper API repository or another open-source implementation.

When you set up your API, you will need to download a Whisper model. This script is configured to use a whisper-large model, so make sure your API server has that model downloaded and loaded.

🚀 Getting Started
Follow these steps to get the project up and running.

1. Prerequisites (What you need first)
Python 3.8+

FFmpeg: A tool for handling audio and video. If you don't have it, you can install it from its official website.

A Hugging Face Account: To get an API token for the speaker diarization model. You can sign up for free at huggingface.co.

Your Own Whisper API: As mentioned above, you need a running Whisper API service with the large model loaded.

2. Installation
Clone this repository and install the required Python packages:

# Clone the repository
git clone <your-repository-url>
cd <your-repository-name>

# Install Python libraries
pip install -r requirements.txt

(If a requirements.txt file is not available, you can install the packages from the top of the youtube_processing_pipeline.py script.)

3. Configuration
Open the youtube_processing_pipeline.py script and edit the CONFIGURATION section at the top:

YOUTUBE_URLS: Add the list of YouTube videos you want to process.

HUGGINGFACE_TOKEN: Paste your access token from your Hugging Face account.

WHISPER_API_URL: Change this to the URL of your running Whisper API service.

LLM_API_KEY: Add the API key for your Whisper service if it requires one.

▶️ How to Run
Once everything is set up and configured, just run the script from your terminal:

python youtube_processing_pipeline.py

The script will start downloading the videos, processing them, and you'll see the progress in your terminal.

📁 What You Get (The Output)
After the script finishes, you will find a new directory named youtube_pipeline. Inside youtube_pipeline/2_processed_output, you will find a .zip file for each video you processed.

Unzip these files, and inside you will find:

metadata.csv: The main transcript file with columns for start time, end time, speaker, and the text.

A segments folder: Contains all the small audio clips that were sent to the Whisper API.
