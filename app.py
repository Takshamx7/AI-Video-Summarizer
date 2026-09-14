#Import the streamlit library 
#we use Streamlit to create a web interface for our AI Video Summarizer application
import streamlit as st

#Import the OpenCV library to read and process video files
import cv2

#import subprocess to run FFMPEG commands for video processing
import subprocess

#import imageio_ffmopeg to find the FFmpeg executable
import imageio_ffmpeg

#Display the main title of our application
st.title("AI Video Summarizer")

#Display a brief description of the application
st.write("Welcome to the AI Video Summarizer! Summarize any video using AI.")

#display a smaller heading for the video input section
st.subheader("choose your video")

#create a text box where the user can enter a video URL
#whatever the user enters will be stored in the 'url' variable
url = st.text_input("Enter the URL of the video you want to summarize:")

#crete a file upload button
#The user can upload a video from their computer
#we currently allow four common video formats: mp4, mov, avi, and mkv
video_file =st.file_uploader("Or upload a video", type =["mp4","mov","avi","mkv"])

#create a button that the user can click to summarize the video
#The code inside this block will only run when the user clicks the button
if st.button("Summarize Video"):

    #check whether the user has provided a video URL or uploaded a video file
    if url:

        #display the URL entered by the user
        st.success("video URL received!", url)

    #check whether the user has uploaded a video file 
    elif video_file:

        #save the uploaded video file as a file
        with open(video_file.name,"wb") as f:
            #write the uploaded video data into the file

            f.write(video_file.getbuffer())
        #display a success message indicating that the video file has been received
        st.success(f"video_file {video_file.name} received!")

        #open the saved video file using OpenCV
        video = cv2.VideoCapture(video_file.name)

        #check whether Open Cv was able to open the file
        if video.isOpened():

            #get the video's frame rate (frames per second)
            fps = video.get(cv2.CAP_PROP_FPS)

            #calculate the video's duration in seconds
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps

            #display a success message
            st.success(f"video opened successfully: {video_file.name}")

            #display the basic information about the video: duration, total frames, and frames per second
            st.write(f"Duration: {duration:.2f} seconds")
            st.write(f"Total frames: {total_frames}")
            st.write(f"FPS: {fps:.2f}")

            #decide how many frames we wnat to extract
            number_of_frames = 5

            #create a heading for the extracred frames
            st.subheader("Video frames")

            #loop through the numer of frames we want 
            for i in range(number_of_frames):

                #calulate the timestamp for this frame
                #subtract a small amount so we don't requesrt a frame exactly at the end
                timestamp = ((duration - 0.1) /(number_of_frames - 1 )) * i

                #move the video to that timestamp in milliseconds
                video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

                #read the frames from the video
                success,frames = video.read()

                #check whether the frames was successfully read
                if success:

                    #convert the frames from BGR to RGB
                    frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

                    #display the timestamp of the frames
                    st.write(f"Frame at {timestamp:.2f} seconds")

                    #display the extraced frame
                    st.image(frames)

                else:

                    #show an error if the frame could not be read
                    st.warning(f"Could not read frame at {timestamp:.2f} seconds")
        else:
            #display an error message if OpenCV was unable to open the video file
            st.error("Error: Unable to open the video file.")

        #close the video after reading its information
        video.release()

        #audio extraction

        #get the location of the FFmpeg executable using imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        #create the name of the audio file
        audio_file = "audio.wav"

        #extract audio from the video using FFmpeg
        result = subprocess.run(
            [
                #tell Python which FFmpeg program to use
                ffmpeg_path,
                #tell FFmpeg which video to process
                "-i", video_file.name,
                #remove the video stream and keep only audio
                "-vn",
                #convert the audio to WAV format with PCM 16-bit little-endian encoding
                "-acodec", "pcm_s16le",
                #set the audio sample rate to 44100 Hz
                "-ar", "44100",
                #convert the audio to mono (single channel)
                "-ac", "1",
                #replace the output file if it already exists
                "-y",
                #name of the extracted audio file
                audio_file,
            ],
            #hide FFmpeg's detailed output from the streamlit terminal to keep the interface clean
            capture_output=True,
            #return an error code instead of stopping the program
            check=False,
        )

        #check whether FFmpeg successfully extracted the audio
        if result.returncode == 0:
            #tell the user that audio extracted was successful
            st.success(f"Audio extracted successfully:{audio_file}")

        else:
            #tell the user that this video may not contain audio
            st.warning("No usable audio was found in this video.")

    #If neither a URL nor a video file is provided, display a warning message
    else:
        st.warning("Please provide a video URL or upload a video file to summarize.")