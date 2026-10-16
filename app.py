# Import Streamlit to create the web interface
import streamlit as st

# Import OpenCV to read and process video files
import cv2

# Import subprocess to run FFmpeg commands
import subprocess

# Import imageio_ffmpeg to find the FFmpeg executable
import imageio_ffmpeg

# Import base64 to convert video frames into API-compatible image data
import base64

# Import requests to communicate with the OpenRouter API
import requests

# Import os to read the API key from the .env file
import os

# Import yt_dlp for downloading YouTube videos
import yt_dlp

# Import Faster-Whisper to convert speech into text
from faster_whisper import WhisperModel

# Import load_dotenv to load variables from the .env file
from dotenv import load_dotenv

# ============================================================
# DOWNLOAD FROM VIDEO URL
# ============================================================


def download_video_from_url(url):
    """
    Download a video from either:
    1. A direct video URL such as an MP4 URL
    2. A YouTube URL

    Returns:
        The local path of the downloaded video.
        Returns None if the download fails.
    """

    # Remove unnecessary spaces from the URL
    url = url.strip()

    # Create a folder for downloaded videos
    download_folder = "downloads"

    # Create the folder if it does not already exist
    os.makedirs(download_folder, exist_ok=True)

    # Check whether the URL looks like a YouTube URL
    if "youtube.com" in url or "youtu.be" in url:

        try:
            # Configure yt-dlp for downloading the video
            ydl_options = {
                # Save the downloaded video inside the downloads folder
                "outtmpl": os.path.join(
                    download_folder,
                    "%(id)s.%(ext)s"
                ),

                # Select the best available video and audio
                "format": "bestvideo+bestaudio/best",

                # Merge separate video and audio streams into MP4
                "merge_output_format": "mp4",

                # Keep terminal output quiet
                "quiet": True,

                # Show errors if something goes wrong
                "no_warnings": False,
            }

            # Create the yt-dlp downloader
            with yt_dlp.YoutubeDL(ydl_options) as ydl:

                # Get information about the YouTube video
                info = ydl.extract_info(
                    url,
                    download=False
                )

                # Download the video
                ydl.download([url])

                # Get the video ID
                video_id = info["id"]

                # Build the expected downloaded file path
                video_path = os.path.join(
                    download_folder,
                    f"{video_id}.mp4"
                )

                # Check whether the file actually exists
                if os.path.exists(video_path):

                    # Return the local video path
                    return video_path

                # If the expected MP4 was not found,
                # search the download folder for the downloaded file.
                for filename in os.listdir(download_folder):

                    # Check whether the filename starts with the video ID
                    if filename.startswith(video_id + "."):

                        # Return the discovered file path
                        return os.path.join(
                            download_folder,
                            filename
                        )

                # Return None if no downloaded file was found
                return None

        except Exception as e:

            # Display the error in the Streamlit application
            st.error(
                f"YouTube video download failed: {e}"
            )

            # Return None to indicate failure
            return None


    # ============================================================
    # DIRECT VIDEO URL
    # ============================================================

    else:

        try:
            # Send a request to the video URL
            response = requests.get(
                url,
                stream=True,
                timeout=60
            )

            # Stop if the server returned an error
            response.raise_for_status()

            # Create a local filename for the downloaded video
            video_path = os.path.join(
                download_folder,
                "downloaded_video.mp4"
            )

            # Open the local file in binary write mode
            with open(video_path, "wb") as file:

                # Download the video in small chunks
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    # Write the downloaded chunk to the file
                    if chunk:
                        file.write(chunk)

            # Check that the downloaded file exists
            if os.path.exists(video_path):

                # Return the local video path
                return video_path

            # Return None if the file was not created
            return None

        except Exception as e:

            # Display the download error
            st.error(
                f"Video URL download failed: {e}"
            )

            # Return None when downloading fails
            return None

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load variables stored inside the .env file
load_dotenv()

# Read the OpenRouter API key from the environment
openrouter_api_key = os.getenv("openrouter_api_key")

# ============================================================
# FRAME SHARPNESS CALCULATION
# ============================================================

def calculate_sharpness(frame):
    """
    Calculate how sharp a video frame is.

    Higher Laplacian variance generally means the frame
    contains clearer edges and less motion blur.
    """

    # Convert the frame from BGR to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Calculate the Laplacian of the image
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    # Return the variance as the sharpness score
    return laplacian.var()

# ============================================================
# SMART AUTOMATIC KEY-FRAME EXTRACTION
# ============================================================

def extract_key_frames(
    video_path,
    duration,
    sample_interval=0.5,
    change_threshold=0.10,
    minimum_gap=1.5,
    fine_scan_duration=0.6,
    fine_scan_interval=0.1,
    maximum_key_frames=20
):
    """
    Detect meaningful visual changes and then search a short
    period after each change to find the clearest frame.

    This prevents transition or motion-blurred frames from
    becoming the selected key frame.
    """

    # Open the video using OpenCV
    video = cv2.VideoCapture(video_path)

    # Create a list to store the final key frames
    key_frames = []

    # Stop if the video cannot be opened
    if not video.isOpened():
        return key_frames

    # Store the previous frame used for change detection
    previous_gray = None

    # Store the timestamp of the previous selected key frame
    last_key_frame_time = -minimum_gap

    # Start scanning from the beginning
    current_timestamp = 0.0

    # --------------------------------------------------------
    # COARSE SCAN
    # --------------------------------------------------------

    # Scan through the video at regular intervals
    while current_timestamp < duration:

        # Move to the current timestamp
        video.set(
            cv2.CAP_PROP_POS_MSEC,
            current_timestamp * 1000
        )

        # Read the frame
        success, frame = video.read()

        # Move to the next sample if the frame could not be read
        if not success:
            current_timestamp += sample_interval
            continue

        # Resize the frame for faster comparison
        small_frame = cv2.resize(
            frame,
            (160, 90)
        )

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2GRAY
        )

        # Apply a small blur to reduce tiny pixel noise
        gray_frame = cv2.GaussianBlur(
            gray_frame,
            (5, 5),
            0
        )

        # The first frame is always a useful starting point
        if previous_gray is None:

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Store the first frame
            key_frames.append(
                {
                    "timestamp": current_timestamp,
                    "frame": rgb_frame,
                    "change_score": 1.0,
                    "sharpness": calculate_sharpness(frame)
                }
            )

            # Remember when the first key frame was saved
            last_key_frame_time = current_timestamp

            # Store the current frame for future comparison
            previous_gray = gray_frame

            # Move to the next sample
            current_timestamp += sample_interval
            continue

        # ----------------------------------------------------
        # CALCULATE PIXEL CHANGE
        # ----------------------------------------------------

        # Calculate the absolute difference between frames
        difference = cv2.absdiff(
            gray_frame,
            previous_gray
        )

        # Calculate the average amount of visual change
        change_score = (
            difference.mean() / 255.0
        )

        # Check whether the visual change is large enough
        significant_change = (
            change_score >= change_threshold
        )

        # Check whether enough time has passed
        # since the previous selected key frame
        enough_time_passed = (
            current_timestamp
            - last_key_frame_time
            >= minimum_gap
        )

        # ----------------------------------------------------
        # CHANGE DETECTED
        # ----------------------------------------------------

        if (
            significant_change
            and enough_time_passed
        ):

            # Search for a clear frame immediately after
            # the detected visual change
            best_frame = None

            # Store the sharpness of the best frame
            best_sharpness = -1

            # Store the timestamp of the best frame
            best_timestamp = current_timestamp

            # Start the fine scan slightly after the change
            fine_timestamp = (
                current_timestamp
                + 0.05
            )

            # Calculate when the fine scan should stop
            fine_end = min(
                current_timestamp
                + fine_scan_duration,
                duration - 0.01
            )

            # ------------------------------------------------
            # FINE SCAN
            # ------------------------------------------------

            # Check several frames immediately after the change
            while fine_timestamp <= fine_end:

                # Move to the fine-scan timestamp
                video.set(
                    cv2.CAP_PROP_POS_MSEC,
                    fine_timestamp * 1000
                )

                # Read the frame
                fine_success, fine_frame = (
                    video.read()
                )

                # Continue if this frame could not be read
                if not fine_success:
                    fine_timestamp += fine_scan_interval
                    continue

                # Calculate how sharp the frame is
                sharpness = calculate_sharpness(
                    fine_frame
                )

                # Keep the sharpest frame found so far
                if sharpness > best_sharpness:

                    # Update the best sharpness score
                    best_sharpness = sharpness

                    # Store the best frame
                    best_frame = fine_frame.copy()

                    # Store its timestamp
                    best_timestamp = fine_timestamp

                # Move to the next fine-scan frame
                fine_timestamp += fine_scan_interval

            # ------------------------------------------------
            # SAVE BEST FRAME
            # ------------------------------------------------

            # Make sure a frame was successfully found
            if best_frame is not None:

                # Convert BGR to RGB
                best_frame_rgb = cv2.cvtColor(
                    best_frame,
                    cv2.COLOR_BGR2RGB
                )

                # Store the clearest frame after the change
                key_frames.append(
                    {
                        "timestamp": best_timestamp,
                        "frame": best_frame_rgb,
                        "change_score": change_score,
                        "sharpness": best_sharpness
                    }
                )

                # Remember the timestamp
                last_key_frame_time = best_timestamp

                # Stop if we reached the maximum number
                # of allowed key frames
                if len(key_frames) >= maximum_key_frames:
                    break

        # Store the current frame for the next comparison
        previous_gray = gray_frame

        # Move to the next coarse-scan timestamp
        current_timestamp += sample_interval

    # Release the video
    video.release()

    # Return the automatically selected key frames
    return key_frames

# ============================================================
# APPLICATION TITLE
# ============================================================

# Display the main title of the application
st.title("AI Video Summarizer")

# Display a short description of the application
st.write(
    "Welcome to the AI Video Summarizer! "
    "Summarize videos using audio and visual AI."
)


# ============================================================
# VIDEO INPUT
# ============================================================

# Display the video input heading
st.subheader("Choose your video")


# Create a text box for entering a video URL
url = st.text_input(
    "Enter the URL of the video you want to summarize:"
)


# Create a file uploader for local videos
# We currently support four common video formats
video_file = st.file_uploader(
    "Or upload a video",
    type=["mp4", "mov", "avi", "mkv"]
)


# ============================================================
# SUMMARIZE BUTTON
# ============================================================

# Run the following code when the user clicks the button
if st.button("Summarize Video"):

    # --------------------------------------------------------
    # VIDEO URL
    # --------------------------------------------------------

    # Check whether the user provided a video URL
    if url:

        # Tell the user that the URL was received
        st.info("🔗 Video URL received. Downloading video...")

        # Download the video from the provided URL
        downloaded_video_path = download_video_from_url(url)

        # Check whether the video was downloaded successfully
        if downloaded_video_path:

            # Tell the user that downloading succeeded
            st.success(
                f"✅ Video downloaded successfully: "
                f"{downloaded_video_path}"
            )

            # Display the downloaded video in Streamlit
            st.video(downloaded_video_path)

        else:

            # Tell the user that downloading failed
            st.error("❌ Unable to download the video.")

    # --------------------------------------------------------
    # LOCAL VIDEO FILE
    # --------------------------------------------------------

    elif video_file:

        # Save the uploaded video to the project folder
        with open(video_file.name, "wb") as f:

            # Write the uploaded video data into the file
            f.write(video_file.getbuffer())


        # Display a success message
        st.success(
            f"Video file received: {video_file.name}"
        )


        # ====================================================
        # OPEN VIDEO
        # ====================================================

        # Open the uploaded video using OpenCV
        video = cv2.VideoCapture(
            video_file.name
        )


        # Check whether OpenCV opened the video
        if video.isOpened():

            # Get the video's frame rate
            fps = video.get(
                cv2.CAP_PROP_FPS
            )

            # Get the total number of frames
            total_frames = video.get(
                cv2.CAP_PROP_FRAME_COUNT
            )


            # Check whether FPS is valid
            if fps > 0:

                # Calculate the video duration
                duration = total_frames / fps

            else:

                # Stop processing if FPS is invalid
                st.error(
                    "Could not determine video FPS."
                )

                video.release()

                st.stop()


            # Display video information
            st.success(
                f"Video opened successfully: "
                f"{video_file.name}"
            )

            st.write(
                f"Duration: {duration:.2f} seconds"
            )

            st.write(
                f"Total frames: {total_frames}"
            )

            st.write(
                f"FPS: {fps:.2f}"
            )


            # ============================================================
            # SMART AUTOMATIC KEY-FRAME EXTRACTION
            # ============================================================

            # Display the key-frame extraction heading
            st.subheader("🔍 Automatic Key Frames")

            # Tell the user that the video is being analyzed
            st.write(
                "Detecting visual changes and selecting "
                "clear frames..."
            )

            # Automatically detect visual changes and select
            # the clearest frame after each change
            frame_data = extract_key_frames(
                video_file.name,

                # Total video duration
                duration,

                # First-pass visual scan every 0.5 seconds
                sample_interval=0.5,

                # Minimum pixel-change threshold
                change_threshold=0.10,

                # Minimum gap between selected key frames
                minimum_gap=1.5,

                # Search for a clear frame for 0.6 seconds
                # after detecting a visual change
                fine_scan_duration=0.6,

                # Check a frame every 0.1 seconds during
                # the fine scan
                fine_scan_interval=0.1,

                # Maximum number of key frames
                maximum_key_frames=20
            )


            # Check whether key frames were detected
            if frame_data:

                # Display the number of detected frames
                st.success(
                    f"Detected {len(frame_data)} "
                    f"clear key frames."
                )

                # Display the selected frames
                for index, item in enumerate(frame_data):

                    # Display the key-frame number and timestamp
                    st.write(
                        f"🔹 Key Frame {index + 1}: "
                        f"{item['timestamp']:.2f}s"
                    )

                    # Display the selected frame
                    st.image(
                        item["frame"]
                    )

            else:

                # Tell the user if no frames were detected
                st.error(
                    "No key frames could be detected."
                )


        else:

            # Display an error if OpenCV could not open
            # the uploaded video
            st.error(
                "Error: Unable to open the video file."
            )


        # Release the OpenCV video object
        video.release()


        # ====================================================
        # AUDIO EXTRACTION
        # ====================================================

        # Find the FFmpeg executable
        ffmpeg_path = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )


        # Name of the extracted audio file
        audio_file = "audio.wav"


        # Run FFmpeg to extract audio
        result = subprocess.run(

            [
                # FFmpeg executable
                ffmpeg_path,

                # Input video
                "-i",
                video_file.name,

                # Ignore video stream
                "-vn",

                # Use PCM 16-bit WAV audio
                "-acodec",
                "pcm_s16le",

                # Set audio sample rate
                "-ar",
                "44100",

                # Convert audio to mono
                "-ac",
                "1",

                # Replace existing output file
                "-y",

                # Output audio file
                audio_file,
            ],

            # Hide detailed FFmpeg output
            capture_output=True,

            # Don't stop Python automatically on FFmpeg error
            check=False,
        )


        # Check whether audio extraction succeeded
        if result.returncode == 0:

            # Display successful extraction
            st.success(
                f"Audio extracted successfully: "
                f"{audio_file}"
            )


            # =================================================
            # SPEECH TO TEXT
            # =================================================

            # Display the speech processing heading
            st.subheader("Speech Processing")


            # Load the Faster-Whisper model
            # Base gives a reasonable balance between
            # speed and transcription accuracy
            model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8"
            )


            # Transcribe the extracted audio
            # Whisper provides start/end timestamps
            segments, info = model.transcribe(

                # Audio file to transcribe
                audio_file,

                # Beam search size
                beam_size=5,

                # Ignore sections without speech
                vad_filter=True
            )


            # Convert the generator into a list
            # This allows us to use the segments multiple times
            segments = list(segments)


            # Display detected language
            st.write(
                f"Detected language: "
                f"{info.language}"
            )


            # =================================================
            # CHECK SPEECH
            # =================================================

            # Check whether speech was detected
            if segments:

                # Display transcript heading
                st.subheader(
                    "Timestamped Transcript"
                )


                # Display every speech segment
                for segment in segments:

                    # Display start, end and recognized text
                    st.write(
                        f"{segment.start:.2f}s → "
                        f"{segment.end:.2f}s — "
                        f"{segment.text}"
                    )


                # =================================================
                # AUDIO + VISUAL MATCHING
                # =================================================

                # Display matching heading
                st.subheader(
                    "Audio + Visual Matching"
                )


                # Create a list to store matched information
                matched_data = []


                # Process every speech segment
                for segment in segments:

                    # Calculate the middle of the speech segment
                    speech_timestamp = (
                        segment.start +
                        segment.end
                    ) / 2


                    # Make sure frames are available
                    if frame_data:

                        # Find the closest extracted frame
                        closest_frame = min(

                            frame_data,

                            key=lambda frame:
                            abs(
                                frame["timestamp"]
                                - speech_timestamp
                            )
                        )


                        # Store the matched information
                        matched_data.append(
                            {
                                "speech_start":
                                    segment.start,

                                "speech_end":
                                    segment.end,

                                "text":
                                    segment.text.strip(),

                                "frame_timestamp":
                                    closest_frame[
                                        "timestamp"
                                    ],

                                "frame":
                                    closest_frame[
                                        "frame"
                                    ]
                            }
                        )


                # =================================================
                # DISPLAY MATCHED INFORMATION
                # =================================================

                # Display each audio + visual pair
                for item in matched_data:

                    # Display speech timing
                    st.write(
                        f"Speech: "
                        f"{item['speech_start']:.2f}s → "
                        f"{item['speech_end']:.2f}s"
                    )


                    # Display transcript
                    st.write(
                        f"Text: "
                        f"{item['text']}"
                    )


                    # Display matched frame timestamp
                    st.write(
                        f"Closest frame: "
                        f"{item['frame_timestamp']:.2f}s"
                    )


                    # Display matched frame
                    st.image(
                        item["frame"]
                    )


                # =================================================
                # OPENROUTER MULTIMODAL AI
                # =================================================

                st.subheader(
                    "🧠 AI Audio + Visual Understanding"
                )


                # Check whether the OpenRouter API key exists
                if not openrouter_api_key:

                    # Tell the user that the key is missing
                    st.error(
                        "OpenRouter API key was not found. "
                        "Check your .env file."
                    )

                # Continue only when the key exists
                else:

                    # ---------------------------------------------
                    # LIMIT THE FIRST AI TEST
                    # ---------------------------------------------

                    # Use the first 5 matched segments for now.
                    # This keeps the free API usage controlled.
                    ai_items = matched_data[:5]


                    # Check whether matched data exists
                    if ai_items:

                        # Create a list for the AI prompt
                        prompt_parts = []


                        # Create a list for image data
                        image_parts = []


                        # Process every selected matched segment
                        for index, item in enumerate(
                            ai_items
                        ):

                            # Add information about this video
                            # event to the text prompt
                            prompt_parts.append(
                                f"""
Video Event {index + 1}:

Speech timestamp:
{item['speech_start']:.2f}s → {item['speech_end']:.2f}s

Matched visual timestamp:
{item['frame_timestamp']:.2f}s

Spoken text:
{item['text']}
"""
                            )


                            # -----------------------------------------
                            # COMPRESS FRAME BEFORE API REQUEST
                            # -----------------------------------------

                            # Convert RGB frame back to BGR
                            # because OpenCV encoding expects BGR
                            frame_bgr = cv2.cvtColor(
                                item["frame"],
                                cv2.COLOR_RGB2BGR
                            )


                            # Resize the frame to reduce API payload
                            # while keeping enough visual information
                            frame_height, frame_width = (
                                frame_bgr.shape[:2]
                            )


                            # Set a maximum width for the API image
                            max_width = 1000


                            # Resize only when the frame is too wide
                            if frame_width > max_width:

                                # Calculate the new height
                                new_height = int(
                                    frame_height *
                                    max_width /
                                    frame_width
                                )


                                # Resize the frame
                                frame_bgr = cv2.resize(
                                    frame_bgr,
                                    (
                                        max_width,
                                        new_height
                                    )
                                )


                            # Convert the frame to JPEG
                            success, encoded_image = (
                                cv2.imencode(
                                    ".jpg",
                                    frame_bgr,
                                    [
                                        cv2.IMWRITE_JPEG_QUALITY,
                                        80
                                    ]
                                )
                            )


                            # Only continue if JPEG conversion worked
                            if success:

                                # Convert JPEG bytes to Base64
                                base64_image = (
                                    base64.b64encode(
                                        encoded_image.tobytes()
                                    ).decode("utf-8")
                                )


                                # Create a Base64 image URL
                                image_data_url = (
                                    "data:image/jpeg;base64,"
                                    f"{base64_image}"
                                )


                                # Store the image for the API request
                                image_parts.append(
                                    {
                                        "type":
                                            "image_url",

                                        "image_url":
                                            {
                                                "url":
                                                    image_data_url
                                            }
                                    }
                                )


                        # =================================================
                        # CREATE MULTIMODAL PROMPT
                        # =================================================

                        # Combine all event information
                        event_information = "\n".join(
                            prompt_parts
                        )


                        # Create the main instruction for the AI
                        text_instruction = f"""
You are the multimodal analysis engine of an AI Video Summarizer.

Analyze the video using BOTH:
1. The spoken transcript.
2. The visual frames.

Do NOT analyze the transcript and images separately.
Combine them to understand what is actually happening.

Here are the timestamped speech events:

{event_information}

The images attached after this text correspond to the visual
moments described above in the same order.

For each event:
- Explain what the speaker is saying.
- Explain what is visibly happening.
- Combine both pieces of information into one concise event.
- Keep the original timestamps.

Then create a short overall summary of these video events.

Use this structure:

EVENTS
[Timestamp] - Combined explanation

OVERALL SUMMARY
A concise summary of what the video section is about.

IMPORTANT:
Do not invent information that is not supported by either
the transcript or the visual frames.
"""


                        # Create the content list
                        multimodal_content = [

                            # Add the text instructions first
                            {
                                "type": "text",
                                "text": text_instruction
                            }
                        ]


                        # Add all extracted images
                        multimodal_content.extend(
                            image_parts
                        )


                        # =================================================
                        # OPENROUTER REQUEST
                        # =================================================

                        # OpenRouter Chat Completions endpoint
                        openrouter_url = (
                            "https://openrouter.ai/api/v1/"
                            "chat/completions"
                        )


                        # Create request headers
                        headers = {

                            # Authenticate with OpenRouter
                            "Authorization":
                                f"Bearer "
                                f"{openrouter_api_key}",

                            # Tell the API that JSON is being sent
                            "Content-Type":
                                "application/json",
                        }


                        # Create API request body
                        request_data = {

                            # Use the free model router
                            "model":
                                "openrouter/free",

                            # Send text + images
                            "messages": [
                                {
                                    "role": "user",

                                    "content":
                                        multimodal_content
                                }
                            ]
                        }


                        # Show processing message
                        with st.spinner(
                            "🧠 AI is combining audio "
                            "and visual information..."
                        ):

                            # Send the request to OpenRouter
                            response = requests.post(

                                openrouter_url,

                                headers=headers,

                                json=request_data,

                                timeout=180
                            )


                        # =================================================
                        # HANDLE API RESPONSE
                        # =================================================

                        # Display HTTP status for debugging
                        st.write(
                            f"OpenRouter status: "
                            f"{response.status_code}"
                        )


                        # Check whether the request succeeded
                        if response.status_code == 200:

                            # Convert response into Python dictionary
                            result = response.json()


                            # Extract the AI-generated text
                            ai_summary = (
                                result[
                                    "choices"
                                ][0][
                                    "message"
                                ][
                                    "content"
                                ]
                            )


                            # Display the final AI result
                            st.success(
                                "🎉 Multimodal AI analysis "
                                "completed!"
                            )


                            # Display the AI-generated understanding
                            st.markdown(
                                ai_summary
                            )


                        else:

                            # Display API error
                            st.error(
                                "OpenRouter request failed."
                            )


                            # Display detailed error information
                            st.code(
                                response.text
                            )


                    else:

                        # Tell the user that there is no data
                        st.info(
                            "No matched audio + visual "
                            "data is available for AI analysis."
                        )


            else:

                # Tell the user that no speech was found
                st.info(
                    "🔇 No speech detected. "
                    "Visual analysis will still be used."
                )


        else:

            # Tell the user that audio could not be extracted
            st.warning(
                "No usable audio was found in this video."
            )


    # --------------------------------------------------------
    # NO INPUT
    # --------------------------------------------------------

    else:

        # Ask the user to provide a video
        st.warning(
            "Please provide a video URL or "
            "upload a video file to summarize."
        )