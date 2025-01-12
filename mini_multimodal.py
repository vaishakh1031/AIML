import cv2
import pyaudio
import wave
from transformers import pipeline
from PIL import Image
import requests
from dotenv import load_dotenv
import os

# Function to capture an image
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
    # Replace with your own API key and endpoint if using Bing or Google
    api_key = "4894607f209c48dfa6928f7bb4a637b4"
    search_url = "https://api.bing.microsoft.com/global/my-bing-search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": 3}
    
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json()["webPages"]["value"]
        for result in results:
            print(f"- {result['name']}: {result['url']}")
    else:
        print("Error retrieving search results")

# Function to process inputs with AI
def process_inputs(image_path, audio_path, text):
    print("Processing inputs...")

    # Analyze image
    image_analyzer = pipeline("image-classification", model="google/vit-base-patch16-224")
    image_result = image_analyzer(image_path)
    print(f"Image analysis: {image_result}")

    # Analyze audio (Placeholder for actual audio processing)
    audio_result = "Audio processing would involve speech-to-text or audio analysis model here."
    print(f"Audio file at: {audio_path}")

    # Analyze text
    text_analyzer = pipeline("sentiment-analysis")
    text_result = text_analyzer(text)
    print(f"Text analysis: {text_result}")

    # Combine results
    print("\n--- Searching for Related Information ---")
    search_web(f"Image result: {image_result[0]['label']}")
    search_web(f"Text analysis: {text}")

# Main function to run the multimodal application
def main():
    print("=== Multimodal AI System ===")

    # Capture image
    image_path = capture_image()

    # Record audio
    audio_path = record_audio()

    # Accept text input
    text = input("Enter your text input: ")

    # Process inputs
    process_inputs(image_path, audio_path, text)

if __name__ == "__main__":
    main()
