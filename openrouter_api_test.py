# Import os so Python can read variables from the environment
import os

# Import requests so Python can send an HTTP request to OpenRouter
import requests

# Import load_dotenv so Python can load the .env file
from dotenv import load_dotenv


# ---------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------

# Load the variables stored in the .env file
load_dotenv()

# Read the OpenRouter API key
api_key = os.getenv("openrouter_api_key")


# ---------------------------------------
# CHECK API KEY
# ---------------------------------------

# Stop the program if the API key was not found
if not api_key:
    raise ValueError(
        "OpenRouter API key was not found in the .env file."
    )

# Show that the key was found without exposing the actual key
print("✅ OpenRouter API key loaded successfully.")


# ---------------------------------------
# OPENROUTER REQUEST
# ---------------------------------------

# OpenRouter's API endpoint for chat/completion requests
url = "https://openrouter.ai/api/v1/chat/completions"

# Create the HTTP request headers
headers = {
    # Send the API key securely
    "Authorization": f"Bearer {api_key}",

    # Tell OpenRouter that we are sending JSON
    "Content-Type": "application/json",
}

# Create the request body
data = {
    # Use OpenRouter's free model router
    "model": "openrouter/free",

    # Send a very simple test question
    "messages": [
        {
            "role": "user",
            "content": "Explain AI in one simple sentence."
        }
    ],
}


# ---------------------------------------
# SEND REQUEST
# ---------------------------------------

print("🔄 Connecting to OpenRouter...")

# Send the request to OpenRouter
response = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=60
)


# ---------------------------------------
# CHECK RESPONSE
# ---------------------------------------

# Display the HTTP status code
print(f"📡 Status code: {response.status_code}")

# Check whether the request was successful
if response.status_code != 200:

    # Display the error returned by OpenRouter
    print("❌ OpenRouter request failed.")
    print(response.text)

    # Stop the program
    raise SystemExit()


# Convert the response into Python data
result = response.json()


# ---------------------------------------
# DISPLAY AI RESPONSE
# ---------------------------------------

# Extract the AI-generated answer
answer = result["choices"][0]["message"]["content"]

# Display the successful response
print("\n🎉 OpenRouter connection successful!")
print("🤖 AI response:")
print(answer)