# app_streamlit.py
import os
import io
import zipfile
import datetime as dt

import numpy as np
import cv2
import streamlit as st
import pandas as pd
from tomato_pipeline import (
    HsvThresholds, HoughParams,
    process_image, save_report_grid
)

st.set_page_config(page_title="Tomato Maturity – Streamlit UI", layout="wide")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ تنظیمات")

# Image input
st.sidebar.subheader("ورودی تصویر")
default_path = "img/1 (122).jpg"
img_file = st.sidebar.file_uploader("یک تصویر آپلود کن (jpg/png)", type=["jpg","jpeg","png"])
img_path_input = st.sidebar.text_input("یا مسیر فایل:", value=default_path)
run_button = st.sidebar.button("🔍 پردازش کن")

st.sidebar.markdown("---")
st.sidebar.subheader("HSV Thresholds")
red_lower_h = st.sidebar.slider("Red Lower H", 0, 179, 0)
red_upper_h = st.sidebar.slider("Red Upper H", 0, 179, 14)
orange_lower_h = st.sidebar.slider("Orange Lower H", 0, 179, 16)
orange_upper_h = st.sidebar.slider("Orange Upper H", 0, 179, 47)
green_lower_h = st.sidebar.slider("Green Lower H", 0, 179, 8)
green_upper_h = st.sidebar.slider("Green Upper H", 0, 179, 12)

# S و V هم قابل تنظیم:
red_lower_s = st.sidebar.slider("Red Lower S", 0, 255, 54)
red_lower_v = st.sidebar.slider("Red Lower V", 0, 255, 116)
red_upper_s = st.sidebar.slider("Red Upper S", 0, 255, 255)
red_upper_v = st.sidebar.slider("Red Upper V", 0, 255, 255)

orange_lower_s = st.sidebar.slider("Orange Lower S", 0, 255, 72)
orange_lower_v = st.sidebar.slider("Orange Lower V", 0, 255, 69)
orange_upper_s = st.sidebar.slider("Orange Upper S", 0, 255, 255)
orange_upper_v = st.sidebar.slider("Orange Upper V", 0, 255, 205)

green_lower_s = st.sidebar.slider("Green Lower S", 0, 255, 183)
green_lower_v = st.sidebar.slider("Green Lower V", 0, 255, 109)
green_upper_s = st.sidebar.slider("Green Upper S", 0, 255, 217)
green_upper_v = st.sidebar.slider("Green Upper V", 0, 255, 226)

st.sidebar.markdown("---")
st.sidebar.subheader("Hough Circles")
dp = st.sidebar.slider("dp", 0.5, 3.0, 1.56, 0.01)
minDist = st.sidebar.slider("minDist", 10, 500, 200, 1)
param1 = st.sidebar.slider("param1 (Canny High)", 10, 300, 109, 1)
param2 = st.sidebar.slider("param2 (Accumulator)", 1, 200, 64, 1)
minRadius = st.sidebar.slider("minRadius", 5, 400, 97, 1)
maxRadius = st.sidebar.slider("maxRadius", 10, 500, 191, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Edge & Morph")
c_low  = st.sidebar.slider("Canny low", 0, 255, 50, 1)
c_high = st.sidebar.slider("Canny high", 0, 255, 150, 1)
morph_k = st.sidebar.slider("Morph kernel", 1, 21, 5, 2)
dilate_k = st.sidebar.slider("Dilate kernel", 1, 21, 5, 2)
dilate_it = st.sidebar.slider("Dilate iterations", 1, 10, 2, 1)
min_area = st.sidebar.slider("Min contour area", 1, 1000, 50, 1)

st.title("🍅 Tomato Maturity Detection – Streamlit")

# ---------------- Load Image ----------------
def load_image():
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img_bgr, "uploaded_image"
    else:
        if os.path.exists(img_path_input):
            return cv2.imread(img_path_input), os.path.basename(img_path_input)
        return None, None

img_bgr, img_name = load_image()

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("ورودی")
    if img_bgr is not None:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=img_name, use_container_width=True)
    else:
        st.info("یک تصویر آپلود کن یا مسیر صحیح وارد کن.")

with col2:
    st.subheader("خروجی نهایی")
    out_placeholder = st.empty()

# ---------------- Run Pipeline ----------------
if run_button:
    if img_bgr is None:
        st.error("تصویر پیدا نشد.")
        st.stop()

    hsv_th = HsvThresholds(
        red_lower=(red_lower_h, red_lower_s, red_lower_v),
        red_upper=(red_upper_h, red_upper_s, red_upper_v),
        orange_lower=(orange_lower_h, orange_lower_s, orange_lower_v),
        orange_upper=(orange_upper_h, orange_upper_s, orange_upper_v),
        green_lower=(green_lower_h, green_lower_s, green_lower_v),
        green_upper=(green_upper_h, green_upper_s, green_upper_v),
    )
    hough = HoughParams(dp=dp, minDist=minDist, param1=param1, param2=param2,
                        minRadius=minRadius, maxRadius=maxRadius)

    rows, inter = process_image(
        image_bgr=img_bgr,
        hsv_th=hsv_th,
        hough=hough,
        morph_kernel_size=morph_k,
        edge_canny_low=c_low,
        edge_canny_high=c_high,
        dilate_size=dilate_k,
        dilate_iter=dilate_it,
        contour_min_area=min_area,
        circle_radius_min=minRadius,
        circle_radius_max=maxRadius,
    )

    # نمایش خروجی نهایی
    out_placeholder.image(inter["detected"], caption="Detected Circles & Labels", use_container_width=True)

    # جدول نتایج
    if rows:
        df = pd.DataFrame([{
            "Index": r.idx, "x": r.x, "y": r.y, "radius": r.r,
            "mean_hue": round(r.mean_hue, 2), "status": r.status
        } for r in rows])
        st.subheader("خلاصهٔ گوجه‌ها")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("دایره‌ای پیدا نشد. پارامترهای Hough/HSV را تغییر بده.")

    # تب دیباگ برای تصاویر میانی
    with st.expander("🔎 Debug / تصاویر میانی", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.image(inter["mask_red"],    caption="Red Mask", use_container_width=True)
        c2.image(inter["mask_orange"], caption="Orange Mask", use_container_width=True)
        c3.image(inter["mask_green"],  caption="Green Mask", use_container_width=True)

        c4, c5, c6 = st.columns(3)
        c4.image(inter["combined_clean"], caption="Cleaned Combined", use_container_width=True)
        c5.image(inter["filled_mask"],    caption="Filled Mask", use_container_width=True)
        c6.image(inter["filled_mask_rgb"],caption="Masked RGB", use_container_width=True)

        c7, c8, c9 = st.columns(3)
        c7.image(inter["edges"], caption="Edges", use_container_width=True)
        c8.image(inter["dilated_edges"], caption="Dilated Edges", use_container_width=True)
        c9.image(inter["separated_mask_rgb"], caption="Separated on RGB", use_container_width=True)

    # ذخیره خروجی‌ها + دانلود
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(img_name)[0] if img_name else "result"
    out_prefix = f"{base}_{timestamp}"

    detected_path = os.path.join(OUT_DIR, f"{out_prefix}_detected.jpg")
    cv2.imwrite(detected_path, cv2.cvtColor(inter["detected"], cv2.COLOR_RGB2BGR))

    grid_path = os.path.join(OUT_DIR, f"{out_prefix}_report_grid.png")
    save_report_grid(inter, grid_path)

    # ZIP برای دانلود
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{out_prefix}_params.txt", 
            f"HSV={hsv_th}\nHough={hough}\nCanny=({c_low},{c_high})\nMorph={morph_k}\nDilate=({dilate_k}x{dilate_it})\nMinArea={min_area}")
        zf.writestr(f"{out_prefix}_table.csv", df.to_csv(index=False) if rows else "empty")
        # تصاویر:
        zf.write(detected_path, arcname=os.path.basename(detected_path))
        zf.write(grid_path, arcname=os.path.basename(grid_path))

    mem_zip.seek(0)
    st.download_button(
        "⬇️ دانلود خروجی‌ها (ZIP)",
        data=mem_zip,
        file_name=f"{out_prefix}.zip",
        mime="application/zip"
    )

    st.success(f"ذخیره شد: {os.path.abspath(detected_path)} , {os.path.abspath(grid_path)}")
