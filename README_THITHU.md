# Câu1
import cv2
import numpy as np
import random

## Đọc ảnh gốc
image = cv2.imread("a.jpg")

## --- 1. Mean filter ---
mean_filtered = cv2.blur(image, (5, 5))
cv2.imwrite("a_mean_filtered.jpg", mean_filtered)

## --- 2. Xác định biên ảnh (sử dụng Canny) ---
edges = cv2.Canny(image, 100, 200)
cv2.imwrite("a_edges.jpg", edges)

## --- 3. Đổi màu RGB ngẫu nhiên ---
## Đổi từ BGR sang RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
## Tạo thứ tự ngẫu nhiên các kênh
channels = [0, 1, 2]
random.shuffle(channels)
## Hoán đổi kênh màu theo thứ tự ngẫu nhiên
random_rgb_image = rgb_image[:, :, channels]
## Lưu ảnh đã đổi màu
random_bgr_image = cv2.cvtColor(random_rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("a_random_color.jpg", random_bgr_image)

## --- 4. Chuyển sang HSV và tách kênh ---
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)
cv2.imwrite("a_hue.jpg", h)
cv2.imwrite("a_saturation.jpg", s)
cv2.imwrite("a_value.jpg", v)

# Câu 2
import cv2
import numpy as np
import random
import os

## 🔹 Đọc 3 ảnh gốc
image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']
images = []
for file in image_files:
    if not os.path.exists(file):
        print(f"❌ Không tìm thấy file {file}")
        images.append(None)
    else:
        images.append(cv2.imread(file))

## 🔹 Hàm biến đổi 1: Inverse (ảnh âm bản)
def inverse_transform(img):
    return 255 - img

## 🔹 Hàm biến đổi 2: Gamma Correction
def gamma_correction(img):
    gamma = round(random.uniform(0.5, 2.0), 2)
    print(f"Gamma = {gamma}")
    img_norm = img / 255.0
    corrected = np.power(img_norm, gamma)
    return np.uint8(corrected * 255)

## 🔹 Hàm biến đổi 3: Log Transform
def log_transform(img):
    c = round(random.uniform(1.0, 5.0), 2)
    print(f"Log coefficient = {c}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = np.float32(gray) + 1.0
    log_img = c * np.log(img_float)
    log_img = np.uint8(255 * log_img / np.max(log_img))
    return cv2.cvtColor(log_img, cv2.COLOR_GRAY2BGR)

## 🔹 Hàm biến đổi 4: Histogram Equalization
def histogram_equalization(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

## 🔹 Hàm biến đổi 5: Contrast Stretching
def contrast_stretching(img):
    min_val = random.randint(0, 50)
    max_val = random.randint(200, 255)
    print(f'Contrast stretching: min={min_val}, max={max_val}')
    stretched = np.clip((img - min_val) * 255 / (max_val - min_val), 0, 255)
    return np.uint8(stretched)

## 🔹 Hàm biến đổi 6: Adaptive Histogram Equalization (CLAHE)
def adaptive_histogram(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

## 🔹 Từ phím -> hàm
transformations = {
    'I': ('inverse', inverse_transform),
    'G': ('gamma', gamma_correction),
    'L': ('log', log_transform),
    'H': ('histogram', histogram_equalization),
    'C': ('contrast', contrast_stretching),
    'A': ('adaptive', adaptive_histogram)
}

## 🔹 MENU in ra cho người dùng
print("📌 Nhấn phím tương ứng để áp dụng biến đổi ảnh:")
print("I: Inverse")
print("G: Gamma Correction")
print("L: Log Transform")
print("H: Histogram Equalization")
print("C: Contrast Stretching")
print("A: Adaptive Histogram Equalization")

## ✅ Phần dưới chỉ chạy trong môi trường thực thi tương tác (trên máy bạn)
key = input("👉 Nhập phím: ").upper()

if key in transformations:
    method_name, method_func = transformations[key]
    for idx, img in enumerate(images):
        if img is None:
            continue
        result = method_func(img)
        filename = f"output_{method_name}_{idx+1}.jpg"
        cv2.imwrite(filename, result)


# Câu 3
import cv2
import numpy as np
import random
import os
# câu 2

## 🔹 Đọc 3 ảnh gốc
image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']
images = []
for file in image_files:
    if not os.path.exists(file):
        print(f"❌ Không tìm thấy file {file}")
        images.append(None)
    else:
        images.append(cv2.imread(file))

## 🔹 Hàm biến đổi 1: Inverse (ảnh âm bản)
def inverse_transform(img):
    return 255 - img

## 🔹 Hàm biến đổi 2: Gamma Correction
def gamma_correction(img):
    gamma = round(random.uniform(0.5, 2.0), 2)
    print(f"Gamma = {gamma}")
    img_norm = img / 255.0
    corrected = np.power(img_norm, gamma)
    return np.uint8(corrected * 255)

## 🔹 Hàm biến đổi 3: Log Transform
def log_transform(img):
    c = round(random.uniform(1.0, 5.0), 2)
    print(f"Log coefficient = {c}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = np.float32(gray) + 1.0
    log_img = c * np.log(img_float)
    log_img = np.uint8(255 * log_img / np.max(log_img))
    return cv2.cvtColor(log_img, cv2.COLOR_GRAY2BGR)

## 🔹 Hàm biến đổi 4: Histogram Equalization
def histogram_equalization(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

## 🔹 Hàm biến đổi 5: Contrast Stretching
def contrast_stretching(img):
    min_val = random.randint(0, 50)
    max_val = random.randint(200, 255)
    print(f'Contrast stretching: min={min_val}, max={max_val}')
    stretched = np.clip((img - min_val) * 255 / (max_val - min_val), 0, 255)
    return np.uint8(stretched)

## 🔹 Hàm biến đổi 6: Adaptive Histogram Equalization (CLAHE)
def adaptive_histogram(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

## 🔹 Từ phím -> hàm
transformations = {
    'I': ('inverse', inverse_transform),
    'G': ('gamma', gamma_correction),
    'L': ('log', log_transform),
    'H': ('histogram', histogram_equalization),
    'C': ('contrast', contrast_stretching),
    'A': ('adaptive', adaptive_histogram)
}

## 🔹 MENU in ra cho người dùng
print("📌 Nhấn phím tương ứng để áp dụng biến đổi ảnh:")
print("I: Inverse")
print("G: Gamma Correction")
print("L: Log Transform")
print("H: Histogram Equalization")
print("C: Contrast Stretching")
print("A: Adaptive Histogram Equalization")

## ✅ Phần dưới chỉ chạy trong môi trường thực thi tương tác (trên máy bạn)
key = input("👉 Nhập phím: ").upper()

if key in transformations:
    method_name, method_func = transformations[key]
    for idx, img in enumerate(images):
        if img is None:
            continue
        result = method_func(img)
        filename = f"output_{method_name}_{idx+1}.jpg"
        cv2.imwrite(filename, result)
