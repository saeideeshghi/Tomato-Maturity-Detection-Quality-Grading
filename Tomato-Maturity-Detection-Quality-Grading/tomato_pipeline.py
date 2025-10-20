# tomato_pipeline.py
import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# -------------------- Config & Types --------------------
@dataclass
class HsvThresholds:
    red_lower:    tuple = (0, 54, 116)
    red_upper:    tuple = (14, 255, 255)
    orange_lower: tuple = (16, 72, 69)
    orange_upper: tuple = (47, 255, 205)
    green_lower:  tuple = (8, 183, 109)    # توجه: در محدوده HSV اوپن‌سی‌وی (0..179)
    green_upper:  tuple = (12, 217, 226)

@dataclass
class HoughParams:
    dp: float = 1.56
    minDist: int = 200
    param1: int = 109
    param2: int = 64
    minRadius: int = 97
    maxRadius: int = 191

@dataclass
class TomatoResult:
    idx: int
    x: int
    y: int
    r: int
    mean_hue: float
    status: str

# -------------------- Classification --------------------
def classify_tomato(hue_val: float) -> str:
    """
    Classify a tomato's ripeness based on the mean Hue value.
    - Ripe       : Hue in [0, 15]
    - Half-Ripe  : Hue in [16, 24]
    - Unripe     : Otherwise
    (Hue در OpenCV بین 0..179 است)
    """
    if 0 <= hue_val <= 15:
        return "Ripe"
    elif 16 <= hue_val <= 24:
        return "Half-Ripe"
    else:
        return "Unripe"

# -------------------- Core Pipeline --------------------
def process_image(
    image_bgr: np.ndarray,
    hsv_th: HsvThresholds = HsvThresholds(),
    hough: HoughParams = HoughParams(),
    morph_kernel_size: int = 5,
    edge_canny_low: int = 50,
    edge_canny_high: int = 150,
    dilate_size: int = 5,
    dilate_iter: int = 2,
    contour_min_area: int = 50,
    circle_radius_min: int = 90,
    circle_radius_max: int = 200,
):
    """
    تمام مراحل پردازش را اجرا می‌کند و هم نتایج عددی و هم تصاویر میانی را برمی‌گرداند.
    """
    assert image_bgr is not None, "Input image is None."
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # --- Masks
    lower_red    = np.array(hsv_th.red_lower, dtype=np.uint8)
    upper_red    = np.array(hsv_th.red_upper, dtype=np.uint8)
    lower_orange = np.array(hsv_th.orange_lower, dtype=np.uint8)
    upper_orange = np.array(hsv_th.orange_upper, dtype=np.uint8)
    lower_green  = np.array(hsv_th.green_lower, dtype=np.uint8)
    upper_green  = np.array(hsv_th.green_upper, dtype=np.uint8)

    mask_red    = cv2.inRange(image_hsv, lower_red,    upper_red)
    mask_orange = cv2.inRange(image_hsv, lower_orange, upper_orange)
    mask_green  = cv2.inRange(image_hsv, lower_green,  upper_green)

    combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_orange), mask_green)

    # --- Morph
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN,  kernel)

    # --- Fill holes by contours
    filled_mask = cleaned_mask.copy()
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # --- Apply mask
    filled_mask_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=filled_mask)

    # --- Edges + dilation
    filled_mask_gray = cv2.cvtColor(filled_mask_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(filled_mask_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, edge_canny_low, edge_canny_high)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
    dilated_edges = cv2.dilate(edges, dilate_kernel, iterations=dilate_iter)

    separated_mask = np.zeros_like(dilated_edges)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > contour_min_area:
            cv2.drawContours(separated_mask, [contour], -1, 255, thickness=cv2.FILLED)
    separated_mask_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=separated_mask)

    # --- Hough circles
    blur_for_circles = cv2.GaussianBlur(edges, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur_for_circles,
        cv2.HOUGH_GRADIENT,
        dp=hough.dp,
        minDist=hough.minDist,
        param1=hough.param1,
        param2=hough.param2,
        minRadius=hough.minRadius,
        maxRadius=hough.maxRadius,
    )

    image_with_circles = image_rgb.copy()
    filtered_circles = []
    tomato_rows = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if circle_radius_min <= r <= circle_radius_max:
                filtered_circles.append((x, y, r))

    for idx, (x, y, r) in enumerate(filtered_circles, start=1):
        circle_mask = np.zeros((image_hsv.shape[0], image_hsv.shape[1]), dtype=np.uint8)
        cv2.circle(circle_mask, (x, y), r, 255, -1)

        mean_hsv = cv2.mean(image_hsv, mask=circle_mask)
        mean_hue = float(mean_hsv[0])
        status = classify_tomato(mean_hue)

        # Draw circle + labels
        cv2.circle(image_with_circles, (x, y), r, (0, 255, 0), 3)
        text = str(idx)
        tsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        tx = x - tsize[0] // 2
        ty = y + tsize[1] // 2
        cv2.putText(image_with_circles, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

        label_text = f"{status}"
        lsize, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        lx = x - lsize[0] // 2
        ly = y - r - 10
        cv2.putText(image_with_circles, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        tomato_rows.append(TomatoResult(idx, x, y, r, mean_hue, status))

    intermediates = {
        "original_rgb": image_rgb,
        "mask_red": mask_red,
        "mask_orange": mask_orange,
        "mask_green": mask_green,
        "combined_clean": cleaned_mask,
        "filled_mask": filled_mask,
        "filled_mask_rgb": filled_mask_rgb,
        "edges": edges,
        "dilated_edges": dilated_edges,
        "separated_mask_rgb": separated_mask_rgb,
        "detected": image_with_circles,
    }

    return tomato_rows, intermediates

# -------------------- Optional: save grid (for script mode) --------------------
def save_report_grid(intermediates: dict, out_path: str):
    fig, axes = plt.subplots(4, 3, figsize=(18, 28))

    axes[0, 0].imshow(intermediates["original_rgb"]); axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    axes[0, 1].imshow(intermediates["mask_red"], cmap="gray"); axes[0, 1].set_title("Red Mask"); axes[0, 1].axis("off")
    axes[0, 2].imshow(intermediates["mask_orange"], cmap="gray"); axes[0, 2].set_title("Orange Mask"); axes[0, 2].axis("off")

    axes[1, 0].imshow(intermediates["mask_green"], cmap="gray"); axes[1, 0].set_title("Green Mask"); axes[1, 0].axis("off")
    axes[1, 1].imshow(intermediates["combined_clean"], cmap="gray"); axes[1, 1].set_title("Cleaned Combined"); axes[1, 1].axis("off")
    axes[1, 2].imshow(intermediates["filled_mask"], cmap="gray"); axes[1, 2].set_title("Filled Mask"); axes[1, 2].axis("off")

    axes[2, 0].imshow(intermediates["filled_mask_rgb"]); axes[2, 0].set_title("Masked RGB"); axes[2, 0].axis("off")
    axes[2, 1].imshow(intermediates["edges"], cmap="gray"); axes[2, 1].set_title("Edges"); axes[2, 1].axis("off")
    axes[2, 2].imshow(intermediates["dilated_edges"], cmap="gray"); axes[2, 2].set_title("Dilated Edges"); axes[2, 2].axis("off")

    axes[3, 0].imshow(intermediates["separated_mask_rgb"]); axes[3, 0].set_title("Separated on RGB"); axes[3, 0].axis("off")
    axes[3, 1].imshow(intermediates["detected"]); axes[3, 1].set_title("Detected Circles & Labels"); axes[3, 1].axis("off")
    axes[3, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
