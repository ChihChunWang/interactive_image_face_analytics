import numpy as np
import cv2
import io
from PIL import Image
from typing import Tuple, Dict, Any, Optional

def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    自 Streamlit 上傳元件讀取影像並轉換為 RGB 通道之 NumPy 矩陣
    """
    if uploaded_file is None:
        return None
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def extract_metadata(uploaded_file, image_rgb: np.ndarray) -> Dict[str, Any]:
    """
    安全提取與計算影像基本中繼資訊
    """
    h, w, c = image_rgb.shape
    file_size_bytes = len(uploaded_file.getvalue()) if uploaded_file else 0
    file_size_kb = file_size_bytes / 1024.0
    
    return {
        "filename": getattr(uploaded_file, 'name', 'Unknown'),
        "width": w,
        "height": h,
        "resolution": f"{w} × {h}",
        "channels": c,
        "file_size_kb": file_size_kb,
        "aspect_ratio": f"{w / h:.2f}" if h > 0 else "N/A"
    }