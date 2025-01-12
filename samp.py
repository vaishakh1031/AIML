import os
from dotenv import load_dotenv
import cv2
import pyaudio
import wave
from transformers import pipeline
import requests
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import speech_recognition as sr
# Load environment variables from .env file
load_dotenv()
def analyze_text(text):
    """
    Process text input for sentiment analysis and retrieve related information.
    """
    print("Processing Text...")

    # Perform sentiment analysis using Hugging Face pipeline
    try:
        text_analyzer = pipeline("sentiment-analysis")
        text_result = text_analyzer(text)
        print(f"Text Analysis Result: {text_result}")
        sentiment = text_result[0]["label"]
    except Exception as e:
        print(f"Error during text analysis: {e}")
        return

    # Use search engine to retrieve links
    print("\n--- Searching for Related Information ---")
    search_queries = [
        f"Marketing related to {text}",
        f"Information about {text}",
        f"General insights about {text}"
    ]

    for query in search_queries:
        print(f"\nSearching for: {query}")
        search_web(query)
def record_audio(output_path="captured_audio.wav", record_seconds=5):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Mono channel
    rate = 44100  # 44.1 kHz sampling rate

    p = pyaudio.PyAudio()

    print("Recording...")
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    # Record data for the specified duration
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording complete")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a .wav file
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {output_path}")
    return output_path
def analyze_audio(audio_path):
    """
    Processes the audio file to extract meaningful information.
    :param audio_path: Path to the audio file.
    """
    print("Processing Audio...")

    # Initialize the speech recognizer
    recognizer = sr.Recognizer()

    try:
        # Load the audio file
        with sr.AudioFile(audio_path) as source:
            print("Converting audio to text...")
            audio_data = recognizer.record(source)

        # Perform speech-to-text
        text_result = recognizer.recognize_google(audio_data)
        print(f"Transcription: {text_result}")

        # Search for related information
        print("\n--- Searching for Related Information ---")
        search_web(f"Audio transcription: {text_result}")

    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print(f"Speech Recognition service error: {e}")
    except Exception as e:
        print(f"Error processing audio: {e}")

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

# Function to upload an image file with size limitation and timeout handling
def upload_image(max_size=5):  # max_size in MB
    def choose_file():
        nonlocal file_path
        # Open file dialog and get the selected file path
        file_path = filedialog.askopenfilename(title="Select an Image File",
                                               filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        # Display the selected file path
        if file_path:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB
            if file_size > max_size:
                messagebox.showerror("File Too Large", f"File exceeds {max_size} MB size limit.")
                file_path = None
            else:
                label.config(text=f"Selected File:\n{file_path}")
        else:
            messagebox.showinfo("No Selection", "No file was selected.")

    # Create the Tkinter window
    root = tk.Tk()
    root.title("File Selector")
    root.geometry("500x200")

    # Add a button to open the file dialog
    button = tk.Button(root, text="Choose File", command=choose_file, font=("Arial", 14))
    button.pack(pady=20)

    # Add a label to display the selected file path
    label = tk.Label(root, text="No file selected", font=("Arial", 12), wraplength=450, justify="center")
    label.pack(pady=10)

    file_path = None
    root.mainloop()

    return file_path

# Function to analyze an uploaded image
def analyze_image(image_path):
    print("Processing Image...")
    image_analyzer = pipeline("image-classification", model="google/vit-base-patch16-224")
    image_result = image_analyzer(image_path)
    print(f"Image analysis: {image_result}")

    # Search related info based on image result
    print("\n--- Searching for Related Information ---")
    search_web(f"Image result: {image_result[0]['label']}")
def capture_image(output_file="captured_image.jpg"):
    """
    Captures an image using the default camera.
    :param output_file: File name to save the captured image.
    :return: Path of the saved image or None if capture failed.
    """
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Open the default camera (0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return None

    print("Press 'Space' to capture an image or 'Esc' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture an image.")
            break

        # Display the camera feed
        cv2.imshow("Capture Image", frame)

        # Wait for a key press
        key = cv2.waitKey(1)
        if key == 32:  # Space key to capture
            cv2.imwrite(output_file, frame)
            print(f"Image captured and saved as {output_file}.")
            break
        elif key == 27:  # Esc key to exit
            print("Image capture canceled.")
            output_file = None
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
    return output_file

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
