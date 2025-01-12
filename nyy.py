import os
from dotenv import load_dotenv
import cv2
import pyaudio
import wave
from transformers import pipeline
from PIL import Image
import requests
from tkinter import Tk, filedialog

# Load environment variables from .env file
load_dotenv()

# Function to capture an image from video input
def capture_image(output_path="captured_image.jpg"):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Image")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Image", frame)

        k = cv2.waitKey(1)
        if k % 256 == 32:  # Space key to capture
            cv2.imwrite(output_path, frame)
            print(f"Image saved to {output_path}")
            break

    cam.release()
    cv2.destroyAllWindows()
    return output_path 

# Function to record audio
def record_audio(output_path="captured_audio.wav", record_seconds=5):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()

    print("Recording...")
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording complete")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    return output_path

# Function to search for information
def search_web(query):
    # Retrieve API key from environment variable
    api_key = os.getenv("BING_API_KEY")
    if not api_key:
        print("Error: API key not found. Please check your .env file.")
        return

    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": 3}

    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json().get("webPages", {}).get("value", [])
        for result in results:
            print(f"- {result['name']}: {result['url']}")
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving search results: {e}")

# Function to process image (either capture or upload)
def analyze_image(image_path):
    print("Processing Image...")
    image_analyzer = pipeline("image-classification", model="google/vit-base-patch16-224")
    image_result = image_analyzer(image_path)
    print(f"Image analysis: {image_result}")

    # Search related info based on image result
    print("\n--- Searching for Related Information ---")
    search_web(f"Image result: {image_result[0]['label']}")

# Function to upload an image file
def upload_image():
    print("Uploading Image...")
    root = Tk()
    root.withdraw()  # Hide the Tkinter root window
    file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if not file_path:
        print("No image selected.")
        return None
    print(f"Image uploaded: {file_path}")
    return file_path

# Function to process audio input (audio file)
def analyze_audio(audio_path):
    print("Processing Audio...")
    # Placeholder: Audio processing would involve speech-to-text or audio analysis model
    audio_result = "Audio processing would involve speech-to-text or audio analysis model here."
    print(f"Audio file at: {audio_path}")

    # Search related info based on audio result
    print("\n--- Searching for Related Information ---")
    search_web(f"Audio file analysis result: {audio_result}")

# Function to process text input (sentiment analysis)
def analyze_text(text):
    print("Processing Text...")
    text_analyzer = pipeline("sentiment-analysis")
    text_result = text_analyzer(text)
    print(f"Text analysis: {text_result}")

    # Search related info based on text analysis
    print("\n--- Searching for Related Information ---")
    search_web(f"Text analysis: {text}")

# Main function to run the multimodal application separately
def main():
    print("=== Multimodal AI System ===")

    # Option to process video, audio, or text separately
    choice = input("Choose input type to process (1: Image, 2: Audio, 3: Text): ")

    if choice == '1':
        # Option to capture image or upload an image
        action = input("Would you like to (1) Capture an image or (2) Upload an image file? ")

        if action == '1':
            # Capture image and analyze
            image_path = capture_image()
            analyze_image(image_path)
        elif action == '2':
            # Upload image and analyze
            image_path = upload_image()
            if image_path:
                analyze_image(image_path)
        else:
            print("Invalid choice.")
    elif choice == '2':
        # Record audio and analyze
        audio_path = record_audio()
        analyze_audio(audio_path)
    elif choice == '3':
        # Accept text input and analyze
        text = input("Enter your text input: ")
        analyze_text(text)
    else:
        print("Invalid choice. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()
