#Import the streamlit library 
#we use Streamlit to create a web interface for our AI Video Summarizer application
import streamlit as st

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
    #display the name of the uploaded video file
        st.success("video file received!", video_file.name)

    #If neither a URL nor a video file is provided, display a warning message
    else:
        st.warning("Please provide a video URL or upload a video file to summarize.")