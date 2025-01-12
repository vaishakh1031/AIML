import os
import time
from transformers import pipeline
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import tkinter.messagebox as messagebox

def analyze_image_folder(folder_path):
    """
    Analyzes all images in a folder and generates accuracy and runtime graphs.
    :param folder_path: Path to the folder containing images.
    """
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    # Load the image classification pipeline
    print("Loading image classification model...")
    image_analyzer = pipeline("image-classification", model="google/vit-base-patch16-224")

    # Initialize metrics
    total_images = 0
    correct_predictions = []  # Simulated accuracy for each image
    runtimes = []  # Runtime for each image

    # Process each image in the folder
    print(f"Analyzing images in folder: {folder_path}")
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(image_path) or not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        total_images += 1
        print(f"Processing image: {file_name}")

        # Measure runtime for each image
        start_time = time.time()
        try:
            # Analyze the image
            results = image_analyzer(image_path)
            runtimes.append(time.time() - start_time)

            # Simulated correctness check (replace with ground truth comparison)
            correct = 1 if total_images % 2 == 0 else 0  # Replace with actual ground truth comparison
            correct_predictions.append(correct)
        except Exception as e:
            print(f"Error processing image {file_name}: {e}")
            runtimes.append(0)
            correct_predictions.append(0)

    # Calculate cumulative accuracy
    cumulative_accuracy = [sum(correct_predictions[:i+1]) / (i+1) * 100 for i in range(len(correct_predictions))]

    # Generate accuracy line graph
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, total_images + 1), cumulative_accuracy, marker='o', color='blue', label="Accuracy (%)")
    plt.title("Cumulative Accuracy per Image")
    plt.xlabel("Image Index")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    accuracy_graph = os.path.join(folder_path, "accuracy_line_graph.png")
    plt.savefig(accuracy_graph)
    plt.show()
    print(f"Accuracy line graph saved to {accuracy_graph}")

    # Generate runtime line graph
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, total_images + 1), runtimes, marker='o', color='red', label="Runtime (seconds)")
    plt.title("Runtime per Image")
    plt.xlabel("Image Index")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.legend()
    runtime_graph = os.path.join(folder_path, "runtime_line_graph.png")
    plt.savefig(runtime_graph)
    plt.show()
    print(f"Runtime line graph saved to {runtime_graph}")

    # Summary
    avg_runtime = sum(runtimes) / total_images if total_images > 0 else 0
    overall_accuracy = (sum(correct_predictions) / total_images) * 100 if total_images > 0 else 0
    print("\n--- Summary ---")
    print(f"Total Images Processed: {total_images}")
    print(f"Average Runtime: {avg_runtime:.2f} seconds")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

def select_folder_and_analyze():
    """
    Opens a folder selection dialog and analyzes images in the selected folder.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", True)  # Bring the dialog to the front
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    if folder_path:
        analyze_image_folder(folder_path)
    else:
        messagebox.showinfo("No Folder Selected", "No folder was selected.")

if __name__ == "__main__":
    print("=== Image Folder Analysis ===")
    select_folder_and_analyze()
