import numpy as np
import cv2
import plotly.graph_objects as go
from typing import Tuple, Dict

def get_color_space_channel(image_rgb: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    """
    依據選定模式進行 RGB/YUV 色彩空間轉換與單通道切換
    """
    if mode == "Original (RGB)":
        return image_rgb, "RGB 綜合強度"
    elif mode == "R 通道":
        return image_rgb[:, :, 0], "Red 通道"
    elif mode == "G 通道":
        return image_rgb[:, :, 1], "Green 通道"
    elif mode == "B 通道":
        return image_rgb[:, :, 2], "Blue 通道"
    
    # 轉換為 YUV 色彩空間
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    if mode == "YUV - Y 亮度":
        return image_yuv[:, :, 0], "Y (Luminance)"
    elif mode == "YUV - U 色度":
        return image_yuv[:, :, 1], "U (Chrominance)"
    elif mode == "YUV - V 色度":
        return image_yuv[:, :, 2], "V (Chrominance)"
    
    return image_rgb, "RGB 綜合強度"

def generate_histogram_plot(display_img: np.ndarray, mode: str) -> go.Figure:
    """
    動態繪製對應通道之直方圖（0-255 強度）
    """
    fig = go.Figure()

    if mode == "Original (RGB)":
        colors = ["#FF4B4B", "#00CC96", "#636EFA"]
        names = ["Red", "Green", "Blue"]
        for i in range(3):
            hist, _ = np.histogram(display_img[:, :, i], bins=256, range=(0, 256))
            fig.add_trace(go.Scatter(
                x=np.arange(256), y=hist, name=names[i],
                line=dict(color=colors[i], width=1.5)
            ))
        fig.update_layout(title="RGB 三通道強度分布直方圖")
    else:
        # 單通道 Plotly 柱狀圖
        color_map = {
            "R 通道": "#FF4B4B", "G 通道": "#00CC96", "B 通道": "#636EFA",
            "YUV - Y 亮度": "#888888", "YUV - U 色度": "#AB63FA", "V (Chrominance)": "#FFA15A"
        }
        bar_color = color_map.get(mode, "#3366CC")
        hist, _ = np.histogram(display_img, bins=256, range=(0, 256))
        
        fig.add_trace(go.Bar(
            x=np.arange(256), y=hist, marker_color=bar_color, name=mode
        ))
        fig.update_layout(title=f"{mode} 像素強度分布直方圖")

    fig.update_layout(
        xaxis_title="像素強度 (0–255)",
        yaxis_title="像素數量 (Counts)",
        xaxis=dict(range=[0, 255]),
        margin=dict(l=30, r=30, t=50, b=30),
        height=350,
        hovermode="x unified"
    )
    return fig