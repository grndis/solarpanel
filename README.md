# Solar Panel Crack Detection

This project is a Flask web application that uses a YOLOv8 model to detect cracks in solar panel images. It supports both image uploads and real-time detection via a webcam.

## Features

- **Image Upload:** Upload an image of a solar panel to detect cracks.
- **Live Detection:** Use your webcam for real-time crack detection on solar panels.
- **Environment Configuration:** Uses a `.env` file for managing application settings.
- **Tailwind CSS:** Styled with Tailwind CSS for a modern user interface.

## Project Structure

```
├── .env.example        # Example environment variables
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
├── app.py              # Main Flask application logic
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # HTML template for the web interface
├── uploads/            # Directory for uploaded and processed images (auto-created)
└── yolo.pt             # YOLOv8 model file (ensure this is present or update MODEL_PATH in .env)
```

## Setup

1.  **Clone the repository (if applicable):**

    ```bash
    git clone https://github.com/grndis/solarpanel
    cd solarpanel
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip3 install -r requirements.txt
    ```

4.  **Set up environment variables:**

    - Copy `.env.example` to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Modify the `.env` file with your specific configurations:
      - `FLASK_DEBUG`: `True` for development, `False` for production.
      - `MODEL_PATH`: Path to your YOLO model file (e.g., `yolo.pt`).
      - `DETECTION_CONFIDENCE`: Confidence threshold for detections (e.g., `0.25`).
      - `UPLOAD_FOLDER`: Directory to store uploaded images (e.g., `uploads`).

5.  **Ensure YOLO model file is present:**
    - Place your `yolo.pt` (or other named model) file in the project root or update `MODEL_PATH` in your `.env` file to point to its location.

## Running the Application

1.  **Start the Flask server:**

    ```bash
    python3 app.py
    ```

2.  Open your web browser and navigate to `http://127.0.0.1:5001` (or the address shown in your terminal).

## Usage

- **Image Upload:**

  1.  Click the "Choose File" button to select an image.
  2.  Click "Upload and Detect".
  3.  The original and detected images (with cracks highlighted) will be displayed.

- **Live Detection:**
  1.  Click "Start Live Detection".
  2.  Allow browser access to your camera if prompted.
  3.  The video feed will appear with an overlay showing detected cracks in real-time.
  4.  Click "Stop Live Detection" to end the live feed.

## Notes

- The `uploads` directory will be created automatically if it doesn't exist when you upload an image.
- Ensure your webcam is functional and permissions are granted for live detection.
- The live detection performance might vary based on your system's capabilities and the complexity of the video feed.
