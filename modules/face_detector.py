import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, List, Dict, Any

MODEL_PATH = "blaze_face_short_range.tflite"

def ensure_model_downloaded():
    """自動下載官方輕量級人臉偵測模型檔"""
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, MODEL_PATH)

def detect_faces_mediapipe(image_rgb: np.ndarray, min_confidence: float = 0.5) -> Tuple[np.ndarray, int, List[Dict[str, Any]]]:
    ensure_model_downloaded()
    
    annotated_img = image_rgb.copy()
    h_img, w_img, _ = image_rgb.shape
    face_details = []

    # 配置新版 Tasks FaceDetector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_confidence)
    
    with vision.FaceDetector.create_from_options(options) as detector:
        # 轉換為 MediaPipe Image 物件
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        for idx, detection in enumerate(detection_result.detections):
            score = detection.categories[0].score
            bbox = detection.bounding_box
            
            xmin, ymin = bbox.origin_x, bbox.origin_y
            w, h = bbox.width, bbox.height

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w_img, xmin + w), min(h_img, ymin + h)

            # 繪製邊界框與標籤
            cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            label = f"Face #{idx+1}: {score:.2%}"
            cv2.putText(
                annotated_img, label, (xmin, max(ymin - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

            face_details.append({
                "人臉編號": f"#{idx+1}",
                "信心度 (Confidence)": f"{score * 100:.2f}%",
                "Bounding Box [X, Y, W, H]": f"[{xmin}, {ymin}, {w}, {h}]"
            })

    return annotated_img, len(face_details), face_details