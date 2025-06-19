import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import io
from ultralytics import YOLO
import logging
import cv2
import numpy as np
from dotenv import load_dotenv
import datetime

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = os.getenv("MODEL_PATH", "yolo.pt")
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", 0.11))  
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.80))  

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

if not os.path.exists(MODEL_PATH):
    app.logger.error(
        f"Model file not found at {MODEL_PATH}. Please ensure it's in the correct location."
    )
    model = None
else:
    try:
        model = YOLO(MODEL_PATH)
        app.logger.info(f"YOLO model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        app.logger.error(f"Error loading YOLO model: {e}")
        model = None

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    current_year = datetime.datetime.now().year
    if request.method == "POST":
        if "file" not in request.files:
            app.logger.warning("No file part in the request.")
            return render_template(
                "index.html", error="No file part", current_year=current_year
            )
        file = request.files["file"]
        if file.filename == "":
            app.logger.warning("No selected file.")
            return render_template(
                "index.html", error="No selected file", current_year=current_year
            )
        if file and allowed_file(file.filename):
            if file.filename.startswith("capture-") and file.filename.endswith(".jpg"):
                filename = file.filename
            else:
                filename = secure_filename(file.filename)

            original_filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(original_filepath)
            app.logger.info(f"File {filename} uploaded/captured successfully.")

            crack_percentage = None

            if model is None:
                app.logger.error("YOLO model is not loaded. Cannot perform detection.")
                return render_template(
                    "index.html",
                    error="Model not loaded. Check server logs.",
                    current_year=current_year,
                )

            try:
                results = model(original_filepath, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD)

                if results:
                    output_filename = "detected_" + filename
                    output_filepath = os.path.join(
                        app.config["UPLOAD_FOLDER"], output_filename
                    )

                    img_to_draw_on = cv2.imread(original_filepath)
                    if img_to_draw_on is None:
                        app.logger.error(
                            f"Could not read image for drawing: {original_filepath}"
                        )
                    else:
                        crack_polygons_for_drawing = []
                        if results and results[0].boxes is not None:
                            for box_data in results[0].boxes:
                                if (
                                    hasattr(box_data, "cls")
                                    and int(box_data.cls[0]) == 0
                                ):
                                    xyxy = box_data.xyxy[0].cpu().numpy()
                                    polygon = Polygon(
                                        [
                                            (xyxy[0], xyxy[1]),
                                            (xyxy[2], xyxy[1]),
                                            (xyxy[2], xyxy[3]),
                                            (xyxy[0], xyxy[3]),
                                        ]
                                    )
                                    crack_polygons_for_drawing.append(polygon)

                        if crack_polygons_for_drawing:
                            merged_cracks_for_drawing = unary_union(
                                crack_polygons_for_drawing
                            )
                            if merged_cracks_for_drawing.geom_type == "Polygon":
                                exterior_coords = np.array(
                                    merged_cracks_for_drawing.exterior.coords,
                                    dtype=np.int32,
                                )
                                cv2.polylines(
                                    img_to_draw_on,
                                    [exterior_coords],
                                    isClosed=True,
                                    color=(255, 0, 0),
                                    thickness=10,
                                )
                            elif merged_cracks_for_drawing.geom_type == "MultiPolygon":
                                for poly in merged_cracks_for_drawing.geoms:
                                    exterior_coords = np.array(
                                        poly.exterior.coords, dtype=np.int32
                                    )
                                    cv2.polylines(
                                        img_to_draw_on,
                                        [exterior_coords],
                                        isClosed=True,
                                        color=(255, 0, 0),
                                        thickness=8,
                                    )
                        cv2.imwrite(output_filepath, img_to_draw_on)
                    app.logger.info(
                        f"Detection complete. Output saved to {output_filepath}"
                    )

                    crack_percentage = calculate_crack_percentage_from_results(
                        results, original_filepath
                    )

                    efficiency = None
                    if crack_percentage is not None:
                        efficiency = -0.4579 * crack_percentage + 97.475
                        efficiency = round(efficiency, 2)

                    if filename.startswith("capture-"):
                        return jsonify(
                            original_image=filename,
                            detected_image=output_filename,
                            crack_percentage=crack_percentage,
                            efficiency=efficiency,
                        )
                    else:
                        return render_template(
                            "index.html",
                            original_image=filename,
                            detected_image=output_filename,
                            crack_percentage=crack_percentage,
                            efficiency=efficiency,
                            current_year=current_year,
                        )
                else:
                    app.logger.warning("Detection returned no results.")
                    if filename.startswith("capture-"):
                        return jsonify(
                            error="Detection failed or no objects found.",
                            original_image=filename,
                        )
                    else:
                        return render_template(
                            "index.html",
                            error="Detection failed or no objects found.",
                            original_image=filename,
                            current_year=current_year,
                        )

            except Exception as e:
                app.logger.error(f"Error during detection: {e}")
                if filename.startswith("capture-"):
                    return jsonify(error=f"Error during detection: {str(e)}")
                else:
                    return render_template(
                        "index.html",
                        error=f"Error during detection: {str(e)}",
                        current_year=current_year,
                    )
        else:
            app.logger.warning(f"File type not allowed: {file.filename}")
            return render_template(
                "index.html", error="File type not allowed", current_year=current_year
            )
    return render_template("index.html", current_year=current_year)


from shapely.geometry import Polygon
from shapely.ops import unary_union


def calculate_crack_percentage_from_results(results, image_path):
    """Calculates crack percentage based on YOLO detection results (bounding box areas)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            app.logger.error(
                f"Could not read image at {image_path} for crack calculation."
            )
            return None

        total_image_area = img.shape[0] * img.shape[1]
        crack_polygons = []

        if results and results[0].boxes is not None:
            for box_data in results[0].boxes:
                if hasattr(box_data, "cls") and int(box_data.cls[0]) == 0:
                    xyxy = box_data.xyxy[0].cpu().numpy()
                    polygon = Polygon(
                        [
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            (xyxy[0], xyxy[3]),
                        ]
                    )
                    crack_polygons.append(polygon)

        if not crack_polygons:
            app.logger.info(f"No crack polygons found for {image_path}")
            return 0

        merged_cracks = unary_union(crack_polygons)
        crack_area = merged_cracks.area

        if total_image_area > 0:
            percentage = (crack_area / total_image_area) * 100
            app.logger.info(
                f"Calculated crack percentage: {percentage:.2f}% for image {image_path}"
            )
            return round(float(percentage), 2)
        return 0
    except Exception as e:
        app.logger.error(
            f"Error calculating crack percentage from results for {image_path}: {e}"
        )
        return None


@app.route("/calculate_crack/<filename>", methods=["GET"])
def calculate_crack_route(filename):
    if model is None:
        app.logger.error("YOLO model is not loaded. Cannot perform calculation.")
        return jsonify(error="Model not loaded. Check server logs."), 500

    image_filename = secure_filename(filename)
    image_filepath = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)

    if not os.path.exists(image_filepath):
        app.logger.error(f"Image file not found for calculation: {image_filepath}")
        return jsonify(error="Image file not found."), 404

    try:
        original_image_path_for_calc = image_filepath
        if image_filename.startswith("detected_"):
            original_image_name = image_filename.replace("detected_", "")
            original_image_path_for_calc = os.path.join(
                app.config["UPLOAD_FOLDER"], original_image_name
            )
            if not os.path.exists(original_image_path_for_calc):
                app.logger.error(
                    f"Original image file not found for calculation: {original_image_path_for_calc}"
                )
                return (
                    jsonify(error="Original image file not found for calculation."),
                    404,
                )

        results = model(original_image_path_for_calc, conf=DETECTION_CONFIDENCE)
        crack_percentage = calculate_crack_percentage_from_results(
            results, original_image_path_for_calc
        )

        efficiency = None
        if crack_percentage is not None:
            efficiency = -0.4579 * crack_percentage + 97.475
            efficiency = round(efficiency, 2)
            return jsonify(crack_percentage=crack_percentage, efficiency=efficiency)
        else:
            return jsonify(error="Failed to calculate crack percentage."), 500
    except Exception as e:
        app.logger.error(f"Error during crack calculation for {image_filename}: {e}")
        return jsonify(error=f"Error during crack calculation: {str(e)}"), 500


@app.route("/uploads/<filename>")
def send_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/live_detect", methods=["POST"])
def live_detect():
    if "file" not in request.files:
        return jsonify(error="No file part in request"), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify(error="No selected file"), 400

    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))

            if model is None:
                app.logger.error("YOLO model is not loaded for live detection.")
                return jsonify(error="Model not loaded"), 500

            results = model(pil_image, conf=0.1)

            detections = []
            if results and results[0].boxes is not None:
                for box_data in results[0].boxes:
                    box = box_data.xyxy[0].tolist()
                    conf = float(box_data.conf[0])
                    cls = int(box_data.cls[0])
                    label = model.names[cls]
                    detections.append({"box": box, "confidence": conf, "label": label})
            return jsonify(detections=detections)

        except Exception as e:
            app.logger.error(f"Error during live detection: {e}")
            return jsonify(error=f"Error during live detection: {str(e)}"), 500
    else:
        return jsonify(error="File type not allowed"), 400


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: YOLO model file '{MODEL_PATH}' not found.")
        print(
            "Please make sure the model file is in the same directory as app.py or update MODEL_PATH."
        )
    else:
        print(f"Attempting to load YOLO model from '{MODEL_PATH}'...")
        if model:
            print("YOLO model loaded. Starting Flask app...")
            app.run(host="0.0.0.0", port=5001, debug=True)
        else:
            print(
                "Failed to load YOLO model. Flask app will start but detection will not work."
            )
            app.run(host="0.0.0.0", debug=True)
