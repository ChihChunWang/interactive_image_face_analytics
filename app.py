import streamlit as st
import numpy as np
from modules.image_loader import load_image_from_upload, extract_metadata
from modules.color_analysis import get_color_space_channel, generate_histogram_plot
from modules.spatial_filter import parse_and_validate_kernel, apply_custom_filter, calculate_psnr
from modules.face_detector import detect_faces_mediapipe

# -----------------------------------------------------------------------------
# 頁面主體設定
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="互動式影像處理與人臉分析工具",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 互動式影像處理與人臉分析系統")
st.caption("支援點陣圖解析、多色彩空間與動態直方圖分析、自訂卷積核濾波/PSNR 評估，以及 MediaPipe AI 人臉偵測。")
st.markdown("---")

# -----------------------------------------------------------------------------
# 側邊欄：檔案載入與中繼資訊
# -----------------------------------------------------------------------------
st.sidebar.header("📁 1. 影像載入與設定")
uploaded_file = st.sidebar.file_uploader(
    "上傳點陣圖檔 (支援 PNG, JPG, JPEG, BMP)",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    image_rgb = load_image_from_upload(uploaded_file)
    
    if image_rgb is None:
        st.error("❌ 檔案載入失敗，請確認上傳的圖片格式是否正常！")
        st.stop()

    metadata = extract_metadata(uploaded_file, image_rgb)

    # 側邊欄中繼資訊卡片
    st.sidebar.subheader("📊 影像中繼資訊 (Metadata)")
    st.sidebar.info(
        f"**檔名**：{metadata['filename']}\n\n"
        f"**解析度**：{metadata['resolution']} 像素\n\n"
        f"**長寬比**：{metadata['aspect_ratio']}:1\n\n"
        f"**色彩維度**：{metadata['channels']} 通道 (RGB)\n\n"
        f"**檔案大小**：{metadata['file_size_kb']:.2f} KB"
    )

    # -------------------------------------------------------------------------
    # 主頁籤功能規劃
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "🌈 多色彩空間與直方圖", 
        "⚙️ 空間濾波與 PSNR 量化", 
        "👤 MediaPipe 人臉偵測"
    ])

    # -------------------------------------------------------------------------
    # 功能 1: 色彩空間轉換與同步直方圖
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("色彩空間切換與動態直方圖分析")
        
        mode = st.radio(
            "選擇顯示通道／色彩空間模式：",
            ["Original (RGB)", "R 通道", "G 通道", "B 通道", "YUV - Y 亮度", "YUV - U 色度", "YUV - V 色度"],
            horizontal=True
        )

        display_img, mode_label = get_color_space_channel(image_rgb, mode)
        fig_hist = generate_histogram_plot(display_img, mode)

        col_img, col_chart = st.columns([1, 1])

        with col_img:
            st.image(display_img, caption=f"前視圖：{mode_label}", use_container_width=True)

        with col_chart:
            st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------------------------------------------------
    # 功能 2: 自訂卷積核空間濾波與 PSNR
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("自訂卷積核（Kernel Matrix）空間濾波與 PSNR 計算")

        col_ctrl, col_view = st.columns([1.2, 2])

        with col_ctrl:
            st.markdown("#### 卷積核設定與防呆參數")
            preset = st.selectbox(
                "預設卷積核模板：",
                ["自訂 (Custom)", "銳化 (Sharpen)", "高斯模糊 3x3 (Gaussian)", "浮雕 (Emboss)", "拉普拉斯邊緣 (Laplacian)"]
            )

            default_presets = {
                "銳化 (Sharpen)": "0, -1, 0\n-1, 5, -1\n0, -1, 0",
                "高斯模糊 3x3 (Gaussian)": "1, 2, 1\n2, 4, 2\n1, 2, 1",
                "浮雕 (Emboss)": "-2, -1, 0\n-1, 1, 1\n0, 1, 2",
                "拉普拉斯邊緣 (Laplacian)": "0, 1, 0\n1, -4, 1\n0, 1, 0",
                "自訂 (Custom)": "0, 0, 0\n0, 1, 0\n0, 0, 0"
            }

            kernel_str = st.text_area(
                "二維矩陣 (逗號分隔列，換行分隔行)：",
                value=default_presets[preset],
                height=110
            )

            auto_normalize = st.checkbox("自動正規化權重 (Sum Normalize)", value=True)

            # 解析與防呆檢查
            kernel, warning_msg = parse_and_validate_kernel(kernel_str, auto_normalize)

            if warning_msg:
                st.warning(warning_msg)

            if kernel is not None:
                st.write("**當前生效之卷積核 (Kernel Matrix)：**")
                st.dataframe(kernel)

        with col_view:
            if kernel is not None:
                filtered_img = apply_custom_filter(image_rgb, kernel)
                psnr_val, mse_val = calculate_psnr(image_rgb, filtered_img)

                c1, c2 = st.columns(2)
                c1.image(image_rgb, caption="原始影像 (Original)", use_container_width=True)
                c2.image(filtered_img, caption="濾波後成果 (Filtered)", use_container_width=True)

                st.markdown("---")
                m1, m2 = st.columns(2)
                
                psnr_str = f"{psnr_val:.2f} dB" if psnr_val != float('inf') else "∞ dB (完全無失真)"
                m1.metric(label="📊 峰值信噪比 (PSNR)", value=psnr_str)
                m2.metric(label="📉 均方誤差 (MSE)", value=f"{mse_val:.4f}")
            else:
                st.error("卷積核矩陣格式錯誤，無法計算空間濾波與 PSNR。")

    # -------------------------------------------------------------------------
    # 功能 3: MediaPipe 人臉偵測
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("MediaPipe 深層學習人臉偵測模組")

        col_f_ctrl, col_f_res = st.columns([1, 2])

        with col_f_ctrl:
            confidence_thresh = st.slider(
                "最低判定信心度門檻 (Confidence Threshold)",
                min_value=0.1, max_value=1.0, value=0.5, step=0.05
            )

        annotated_img, face_count, face_details = detect_faces_mediapipe(image_rgb, confidence_thresh)

        with col_f_res:
            st.image(annotated_img, caption=f"偵測標註成果 (共發現 {face_count} 張人臉)", use_container_width=True)
            
            st.markdown("#### 🎯 偵測統計數據表")
            if face_details:
                st.dataframe(face_details, use_container_width=True)
            else:
                st.info("未在當前門檻下偵測到人臉，建議降低信心度門檻再試。")

else:
    st.info("👈 **操作指引**：請點擊左側邊欄上傳圖片檔（PNG, JPG, BMP），開始進行互動分析。")