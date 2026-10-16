# Import os so Python can read the API key from .env
import os

# Import base64 so we can convert the video frame into a format
# that can be sent through the API
import base64

# Import requests so Python can communicate with OpenRouter
import requests

# Import OpenCV so we can open the video and extract a frame
import cv2

# Import load_dotenv so Python can load the .env file
from dotenv import load_dotenv


# ---------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------

# Load variables stored in the .env file
load_dotenv()

# Read the OpenRouter API key
api_key = os.getenv("openrouter_api_key")

# Check whether the API key exists
if not api_key:
    raise ValueError(
        "OpenRouter API key was not found in the .env file."
    )


# ---------------------------------------
# SELECT YOUR TEST VIDEO
# ---------------------------------------

# IMPORTANT:
# Replace this with the path of the video you want to test.
video_path = r"D:\AI Video Summarizer\text sample #2.mp4"


# ---------------------------------------
# OPEN THE VIDEO
# ---------------------------------------

# Open the video using OpenCV
video = cv2.VideoCapture(video_path)

# Check whether OpenCV successfully opened the video
if not video.isOpened():
    raise ValueError(
        "Could not open the video. Check the video path."
    )


# ---------------------------------------
# GET VIDEO INFORMATION
# ---------------------------------------

# Get the total number of frames in the video
total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

# Get the video frame rate
fps = video.get(
    cv2.CAP_PROP_FPS
)

# Calculate the video duration
duration = total_frames / fps


# ---------------------------------------
# SELECT A FRAME
# ---------------------------------------

# Select the middle of the video for this first test
timestamp = duration / 2

# Move OpenCV to the selected timestamp
video.set(
    cv2.CAP_PROP_POS_MSEC,
    timestamp * 1000
)

# Read the frame
success, frame = video.read()

# Release the video file
video.release()

# Check whether the frame was extracted successfully
if not success:
    raise ValueError(
        "Could not extract a frame from the video."
    )


# ---------------------------------------
# CONVERT FRAME TO JPEG
# ---------------------------------------

# Convert the OpenCV frame into JPEG format
success, encoded_image = cv2.imencode(
    ".jpg",
    frame
)

# Check whether JPEG conversion succeeded
if not success:
    raise ValueError(
        "Could not convert the frame into JPEG format."
    )


# ---------------------------------------
# CONVERT IMAGE TO BASE64
# ---------------------------------------

# Convert the JPEG bytes into Base64 text
base64_image = base64.b64encode(
    encoded_image.tobytes()
).decode("utf-8")

# Create a Base64 image data URL
image_data_url = (
    f"data:image/jpeg;base64,{base64_image}"
)


# ---------------------------------------
# OPENROUTER API
# ---------------------------------------

# OpenRouter's Chat Completions endpoint
url = "https://openrouter.ai/api/v1/chat/completions"

# Create the request headers
headers = {
    # Authenticate using the OpenRouter API key
    "Authorization": f"Bearer {api_key}",

    # Tell OpenRouter that we are sending JSON
    "Content-Type": "application/json",
}


# ---------------------------------------
# MULTIMODAL REQUEST
# ---------------------------------------

# Create the request body
data = {
    # Use OpenRouter's free model router
    "model": "openrouter/free",

    # Send text and image together
    "messages": [
        {
            "role": "user",

            # Multimodal content contains both text and image
            "content": [

                # Ask the AI to analyze the visual
                {
                    "type": "text",
                    "text": (
                        "Analyze this video frame. "
                        "Describe what is visibly happening "
                        "in simple words."
                    ),
                },

                # Send the extracted video frame
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


# ---------------------------------------
# SEND REQUEST
# ---------------------------------------

print("🎥 Video opened successfully.")
print(f"⏱️ Video duration: {duration:.2f} seconds")
print(f"🖼️ Testing frame: {timestamp:.2f} seconds")

print("\n🧠 Sending frame to OpenRouter...")

# Send the multimodal request
response = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=120
)


# ---------------------------------------
# CHECK RESPONSE
# ---------------------------------------

# Display the HTTP status code
print(f"📡 Status code: {response.status_code}")


# Check whether OpenRouter returned an error
if response.status_code != 200:

    # Display the server's error message
    print("\n❌ OpenRouter error:")
    print(response.text)

    # Stop the program
    raise SystemExit()


# Convert the successful response to JSON
result = response.json()


# ---------------------------------------
# DISPLAY AI RESPONSE
# ---------------------------------------

# Extract the AI's response text
answer = result["choices"][0]["message"]["content"]

# Display the result
print("\n🎉 MULTIMODAL TEST SUCCESSFUL!")
print("\n👁️ AI's visual analysis:")
print(answer)