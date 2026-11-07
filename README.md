# 互動式影像處理與人臉分析工具 (Interactive Image Processing & Face Analytics Tool)

本專案為一套基於 Python 打造的互動式Web影像分析與處理工具。結合 Streamlit、OpenCV、Plotly與 MediaPipe Tasks API，提供點陣圖解析、多色彩空間與動態直方圖分析、自訂卷積核濾波與 PSNR 量化評估，以及 AI 人臉偵測標註功能。

---

## 核心功能亮點

1. **影像載入與中繼資訊解析**
   - 支援常見點陣圖格式（PNG、JPG/JPEG、BMP）之讀入與即時預覽。
   - 自動提取與呈現影像基本中繼資訊（解析度、長寬比、色彩維度通道數、檔案大小等）。

2. **多色彩空間轉換與同步直方圖分析**
   - **檢視模式**：原始 RGB 綜合檢視、R/G/B 單通道分離、YUV 色彩空間轉換（Y 亮度、U 色度、V 色度）。
   - **動態直方圖**：結合 Plotly 繪製 0–255 像素強度分布直方圖，支援滑鼠懸浮即時數據檢視。

3. **自訂卷積核 (Kernel Matrix) 空間濾波與 PSNR 評估**
   - **動態卷積矩陣**：支援使用者手動輸入任意維度之二維矩陣（如 3×3, 5×5），並預設多種常見濾波器（銳化、高斯平滑、浮雕、拉普拉斯邊緣檢測）。
   - **防呆機制**：內建非方陣提示、偶數維度警示、零矩陣保護與自動正規化（Normalize）除以零防範。
   - **量化指標 (PSNR)**：即時計算並數值化呈現均方誤差 (MSE) 與峰值信噪比 (PSNR, dB)。

4. **MediaPipe AI 人臉偵測模組**
   - 使用 **MediaPipe Tasks Vision API** (`blaze_face_short_range.tflite`)。
   - 提供互動式門檻滑桿（Confidence Threshold），即時視覺化標註人臉邊界框（Bounding Box）與信心度得分，並匯出偵測統計數據表。

---

## 系統架構圖與資料流 (Architecture & Data Flow)

### 1. 模組化架構圖
```text
interactive_image_face_analytics/
├── app.py                   # Streamlit Web GUI 主進入點
├── modules/                 # 核心功能模組 (Decoupled Modules)
│   ├── image_loader.py      # 1. 影像載入與中繼資訊解析
│   ├── color_analysis.py    # 2. 多色彩空間與 Plotly 直方圖繪製
│   ├── spatial_filter.py    # 3. 卷積濾波運算、防呆機制與 PSNR/MSE 計算
│   └── face_detector.py     # 4. MediaPipe Tasks API 人臉偵測標註
├── blaze_face_short_range.tflite # MediaPipe 人臉偵測模型權重檔
├── requirements.txt         # 第三方套件依賴清單
└── README.md                # 專案說明文件
```

## 2. 資料流說明
```text
[使用者上傳圖片] 
       │
       ├──> modules/image_loader.py    ──> 解析 RGB 矩陣與 Metadata
       │
       ├──> modules/color_analysis.py  ──> RGB/YUV 空間轉換 ──> Plotly 繪製 0-255 直方圖
       │
       ├──> modules/spatial_filter.py  ──> 矩陣防呆校驗 ──> OpenCV filter2D ──> 計算 MSE & PSNR(dB)
       │
       └──> modules/face_detector.py   ──> MediaPipe Tasks Detector ──> Bounding Box & 統計表
```

## 3. 核心演算法與數學公式
### 峰值信噪比 (PSNR) 與均方誤差 (MSE)

對於尺寸 $M \times N$ 之原始影像 $I$ 與處理後影像 $K$：

### 1. 均方誤差 (Mean Squared Error, MSE)

$$
MSE = \frac{1}{M \cdot N}
\sum_{i=0}^{M-1}
\sum_{j=0}^{N-1}
[I(i,j)-K(i,j)]^2
$$

### 2. 峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)

$$
PSNR = 10 \cdot \log_{10}
\left(
\frac{MAX_I^2}{MSE}
\right)
$$

其中：

- $MAX_I = 255$（8-bit 灰階影像）
- 當 $MSE = 0$ 時，（即兩影像完全無失真）時，系統將防呆顯示為 $\infty \text{ dB}$。


## 4. 環境建置與安裝步驟 (Installation)
本專案支援使用傳統 pip 或高效能 Python 套件管理器 uv。

### 克隆專案 (Clone Repository)
```text
git clone [https://github.com/ChihChunWang/interactive_image_face_analytics.git](https://github.com/ChihChunWang/interactive_image_face_analytics.git)
cd interactive_image_face_analytics
```

### 環境安裝
1. 使用 uv
```text
uv sync
```


2. 不使用uv
```text
pip install -r requirements.txt
```

### 執行應用程式 (Usage)
1. 使用uv
```text
uv run streamlit run app.py
```

2. 不使用uv
```text
streamlit run app.py
```

執行後，瀏覽器會自動開啟本地存取網址（預設為 http://localhost:8501）。

### 相依套件說明 (requirements.txt)

1. streamlit>=1.30.0：互動式 Web App 圖形介面框架。

2. opencv-python-headless：提供影像處理、空間濾波與繪圖功能（適合雲端無頭環境部署）。

3. numpy>=1.24.0：高效能矩陣運算與數據處理。

4. pillow>=10.0.0：影像檔案讀取與格式轉化。

5. mediapipe>=1.2.0：Google 輕量級 AI 視覺 Tasks 推論引擎。

6. plotly>=5.18.0：動態互動式圖表繪製。

