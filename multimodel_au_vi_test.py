# Import os so Python can read the API key from the .env file
import os

# Import base64 so the video frame can be converted into a format
# that can be sent through the API
import base64

# Import requests so Python can communicate with OpenRouter
import requests

# Import OpenCV for opening the video and extracting a frame
import cv2

# Import Faster-Whisper for converting speech into text
from faster_whisper import WhisperModel

# Import load_dotenv so Python can load variables from the .env file
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load the variables stored in the .env file
load_dotenv()

# Read the OpenRouter API key from the environment
api_key = os.getenv("openrouter_api_key")


# Check whether the API key was loaded successfully
if not api_key:
    raise ValueError(
        "OpenRouter API key was not found in the .env file."
    )


# ============================================================
# VIDEO PATH
# ============================================================

# IMPORTANT:
# Replace this with the path of your test video
video_path = r"D:\AI Video Summarizer\text sample #2.mp4"


# ============================================================
# OPEN VIDEO
# ============================================================

# Open the video using OpenCV
video = cv2.VideoCapture(video_path)


# Check whether the video opened successfully
if not video.isOpened():
    raise ValueError(
        "Could not open the video. Check the video path."
    )


# ============================================================
# GET VIDEO INFORMATION
# ============================================================

# Get the total number of frames in the video
total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

# Get the video's frame rate
fps = video.get(
    cv2.CAP_PROP_FPS
)


# Check that the frame rate is valid
if fps <= 0:
    video.release()

    raise ValueError(
        "Could not determine the video's frame rate."
    )


# Calculate the video duration in seconds
duration = total_frames / fps


# ============================================================
# SELECT TEST TIMESTAMP
# ============================================================

# For this first combined test, use the middle of the video
test_timestamp = duration / 2


# Display video information
print("🎥 Video opened successfully.")
print(f"⏱️ Video duration: {duration:.2f} seconds")
print(
    f"🎯 Testing timestamp: "
    f"{test_timestamp:.2f} seconds"
)


# ============================================================
# EXTRACT FRAME
# ============================================================

# Move OpenCV to the selected timestamp
video.set(
    cv2.CAP_PROP_POS_MSEC,
    test_timestamp * 1000
)


# Read the frame
success, frame = video.read()


# Release the video because we have the frame we need
video.release()


# Check whether the frame was extracted
if not success:
    raise ValueError(
        "Could not extract the video frame."
    )


print("🖼️ Video frame extracted successfully.")


# ============================================================
# CONVERT FRAME TO JPEG
# ============================================================

# Convert the OpenCV frame into JPEG format
success, encoded_image = cv2.imencode(
    ".jpg",
    frame
)


# Check whether JPEG conversion worked
if not success:
    raise ValueError(
        "Could not convert the frame to JPEG."
    )


# ============================================================
# CONVERT IMAGE TO BASE64
# ============================================================

# Convert the JPEG bytes into Base64 text
base64_image = base64.b64encode(
    encoded_image.tobytes()
).decode("utf-8")


# Create a Base64 image data URL
image_data_url = (
    f"data:image/jpeg;base64,{base64_image}"
)


# ============================================================
# LOAD WHISPER MODEL
# ============================================================

print("\n🎙️ Loading Faster-Whisper model...")


# Load the Whisper model
# "base" gives a reasonable balance between speed and accuracy
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)


# ============================================================
# TRANSCRIBE VIDEO AUDIO
# ============================================================

print("📝 Converting speech into timestamped text...")


# Transcribe the video's audio
# Whisper returns speech segments with start/end timestamps
segments, info = model.transcribe(
    video_path,
    beam_size=5,
    vad_filter=True
)


# Convert the segment generator into a list
# This lets us search through the segments multiple times
segments = list(segments)


# Display detected language
print(
    f"🌐 Detected language: "
    f"{info.language}"
)


# ============================================================
# FIND THE CLOSEST SPEECH SEGMENT
# ============================================================

# Store the closest speech segment
closest_segment = None


# Store the smallest timestamp difference found
smallest_difference = float("inf")


# Check every speech segment
for segment in segments:

    # Calculate the middle timestamp of this speech segment
    speech_middle = (
        segment.start + segment.end
    ) / 2

    # Calculate the difference between the speech timestamp
    # and our selected video-frame timestamp
    difference = abs(
        speech_middle - test_timestamp
    )

    # Check whether this is the closest segment so far
    if difference < smallest_difference:

        # Store this segment as the closest one
        closest_segment = segment

        # Store the new smallest difference
        smallest_difference = difference


# ============================================================
# CHECK WHETHER SPEECH WAS FOUND
# ============================================================

if closest_segment:

    # Get the transcript text
    transcript_text = closest_segment.text.strip()

    # Get the beginning of the speech segment
    speech_start = closest_segment.start

    # Get the end of the speech segment
    speech_end = closest_segment.end

    # Display the selected speech segment
    print("\n🎙️ MATCHED AUDIO SEGMENT:")
    print(
        f"⏱️ {speech_start:.2f}s → "
        f"{speech_end:.2f}s"
    )
    print(f"📝 {transcript_text}")

else:

    # If no speech was detected, use a clear message
    transcript_text = (
        "No speech was detected at this point in the video."
    )

    print("\n🔇 No speech segment was found.")


# ============================================================
# PREPARE OPENROUTER REQUEST
# ============================================================

# OpenRouter's API endpoint
url = "https://openrouter.ai/api/v1/chat/completions"


# Create the request headers
headers = {

    # Authenticate using the OpenRouter API key
    "Authorization": f"Bearer {api_key}",

    # Tell the API that we are sending JSON
    "Content-Type": "application/json",
}


# ============================================================
# CREATE MULTIMODAL PROMPT
# ============================================================

# Create the request body
data = {

    # Use OpenRouter's free model router
    "model": "openrouter/free",

    # Send the audio transcript and visual frame together
    "messages": [

        {
            "role": "user",

            # Multimodal content contains text + image
            "content": [

                # ----------------------------------------
                # AUDIO / SPEECH INFORMATION
                # ----------------------------------------

                {
                    "type": "text",

                    "text": (
                        "You are analyzing a video using both "
                        "spoken audio and visual information.\n\n"

                        f"Video timestamp: "
                        f"{test_timestamp:.2f} seconds\n\n"

                        "Spoken transcript from this part "
                        "of the video:\n"
                        f"{transcript_text}\n\n"

                        "Now examine the video frame and "
                        "combine the spoken information with "
                        "the visible information.\n\n"

                        "Explain what is happening in this "
                        "part of the video in 2-3 simple "
                        "sentences."
                    ),
                },


                # ----------------------------------------
                # VISUAL INFORMATION
                # ----------------------------------------

                {
                    "type": "image_url",

                    "image_url": {
                        "url": image_data_url
                    },
                },
            ],
        }
    ],
}


# ============================================================
# SEND MULTIMODAL REQUEST
# ============================================================

print("\n🧠 Sending AUDIO + VISUAL information to OpenRouter...")


# Send the request to OpenRouter
response = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=120
)


# ============================================================
# CHECK API RESPONSE
# ============================================================

# Display the HTTP status code
print(
    f"📡 Status code: "
    f"{response.status_code}"
)


# Check whether the API request failed
if response.status_code != 200:

    # Display the server's error message
    print("\n❌ OpenRouter error:")
    print(response.text)

    # Stop the program
    raise SystemExit()


# Convert the successful response into Python data
result = response.json()


# ============================================================
# GET AI RESPONSE
# ============================================================

# Extract the generated response
answer = result["choices"][0]["message"]["content"]


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

print("\n" + "=" * 60)

print("🎉 AUDIO + VISUAL MULTIMODAL TEST SUCCESSFUL!")

print("=" * 60)

print("\n🎙️ Spoken information:")
print(transcript_text)

print(
    f"\n👁️ Visual timestamp: "
    f"{test_timestamp:.2f} seconds"
)

print("\n🧠 Combined AI understanding:")
print(answer)

print("\n" + "=" * 60)