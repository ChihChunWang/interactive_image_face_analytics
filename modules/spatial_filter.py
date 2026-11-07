import numpy as np
import cv2
import math
from typing import Tuple, Optional

def parse_and_validate_kernel(text_input: str, auto_normalize: bool) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    嚴格解析與防呆校驗自訂二維卷積核陣列
    包含：格式驗證、非方陣檢測、零矩陣保護與正規化除以零防範
    """
    try:
        lines = [line.strip() for line in text_input.strip().split("\n") if line.strip()]
        if not lines:
            return None, "錯誤：卷積核不可以為空值。"

        matrix = []
        for line in lines:
            row = [float(val.strip()) for val in line.split(",") if val.strip() != ""]
            matrix.append(row)

        kernel = np.array(matrix, dtype=np.float32)

        # 防呆 1: 檢查維度是否為 2D 矩陣
        if kernel.ndim != 2:
            return None, "錯誤：輸入必須為二維矩陣。"

        rows, cols = kernel.shape

        # 防呆 2: 非方陣檢測 (N x N)
        if rows != cols:
            return None, f"警告：目前輸入為 {rows}×{cols} 非對稱矩陣，建議改為正方形矩陣 (如 3×3 或 5×5) 以避免邊緣邊界變形。"

        # 防呆 3: 奇數邊長檢測
        if rows % 2 == 0 or cols % 2 == 0:
            return None, f"注意：目前輸入規格為 {rows}×{cols} (偶數維度)。卷積核通常需為奇數 (如 3×3, 5×5) 才能精確定義中心像素點。"

        # 防呆 4: 全零矩陣檢測
        kernel_sum = np.sum(kernel)
        if kernel_sum == 0 and auto_normalize:
            return kernel, "提示：卷積核權重總和為 0，已自動關閉「正規化」以防止除以零 (Divide by Zero) 錯誤。"

        # 正規化計算
        if auto_normalize and kernel_sum != 0:
            kernel = kernel / kernel_sum

        return kernel, None

    except ValueError:
        return None, "錯誤：矩陣中包含無效字元，請確保數字間以逗號 (,) 分隔，各行以換行符號隔開。"
    except Exception as e:
        return None, f"未知錯誤: {str(e)}"

def apply_custom_filter(image_rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    使用 OpenCV filter2D 執行空間濾波運算
    """
    return cv2.filter2D(image_rgb, -1, kernel)

def calculate_psnr(img_orig: np.ndarray, img_filtered: np.ndarray) -> Tuple[float, float]:
    """
    數值化計算 Peak Signal-to-Noise Ratio (PSNR) 與 Mean Squared Error (MSE)
    防呆除以零機制：當 MSE == 0 時回傳 inf
    """
    mse = np.mean((img_orig.astype(np.float64) - img_filtered.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf'), 0.0
    
    max_pixel = 255.0
    psnr = 10.0 * math.log10((max_pixel ** 2) / mse)
    return psnr, float(mse)