Bài 6
import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pyplot as plt
from PIL import Image

lọc ảnh với mean filter
a = iio.imread('baby.jpeg', mode='F')  
k = np.ones((5,5))/25
b = sn.convolve(a, k).astype(np.uint8)
iio.imsave('baby_mean_filter.png', b)
print(b)
plt.imshow(b)
plt.axis('off')
plt.show()

lọc ảnh với median filter
import imageio.v2 as iio
a = iio.imread('baby.jpeg', mode='L')
b = sn.median_filter(a, size=5, mode='reflect')
iio.imsave('baby_median_filter.jpeg', b)
print(b)
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()

lọc ảnh với min filter
import imageio.v2 as iio
a = iio.imread('baby.jpeg', mode='L')  
b = sn.minimum_filter(a, size=5, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
iio.imsave('baby_min_filter.jpeg', b)
print(b)
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()

Bài 7:
import cv2
import os
Thư mục chứa ảnh gốc
input_folder = 'exercise'
Thư mục lưu ảnh kết quả
output_folder = 'exercise_Bai7'
os.makedirs(output_folder, exist_ok=True)
 Duyệt qua từng ảnh trong thư mục
for fname in os.listdir(input_folder):
      Đọc ảnh xám (grayscale)
    input_path = os.path.join(input_folder, fname) # Bỏ qua nếu không đọc được ảnh
      1. Khử nhiễu bằng median filter
    denoised = cv2.medianBlur(img, 5)
     2. Xác định biên bằng Canny
    edges = cv2.Canny(denoised, 100, 200)
     3. Lưu ảnh kết quả
    output_path = os.path.join(output_folder, f"edge_{fname}")
    cv2.imwrite(output_path, edges)


Bài 8:
import cv2
import os
import numpy as np
import random

 Thư mục gốc chứa ảnh
input_folder = 'Exercise'

 Thư mục để lưu ảnh sau khi đổi màu
output_folder = 'Exercise_RGB'
os.makedirs(output_folder, exist_ok=True)

 Duyệt qua từng ảnh trong thư mục
for fname in os.listdir(input_folder):
    input_path = os.path.join(input_folder, fname)
     Đọc ảnh màu
    img = cv2.imread(input_path)
    if img is None:
     1. Khử nhiễu bằng median filter
    denoised = cv2.medianBlur(img, 5)
     2. Sinh hệ số ngẫu nhiên cho từng kênh BGR (giữ trong khoảng 0.5 - 1.5)
    b_scale = random.uniform(0.5, 1.5)
    g_scale = random.uniform(0.5, 1.5)
    r_scale = random.uniform(0.5, 1.5)
     3. Nhân mỗi kênh với hệ số tương ứng
    b, g, r = cv2.split(denoised)
    b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
    g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
    r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
     Gộp lại ảnh đổi màu
    color_shifted = cv2.merge([b, g, r])
     4. Lưu ảnh mới
    output_path = os.path.join(output_folder, f"rgb_{fname}")
    cv2.imwrite(output_path, color_shifted)

Bài 9:
import cv2
import os
import numpy as np
import random
input_folder = 'Exercise'
output_folder = 'Exercise_HSV'
os.makedirs(output_folder, exist_ok=True)
 Tập hợp các giá trị H đã sử dụng để tránh trùng
used_hue_shifts = set()
 Sinh giá trị H ngẫu nhiên không trùng
def get_unique_hue_shift():
    while True:
        shift = random.randint(0, 179)
        if shift not in used_hue_shifts:
            used_hue_shifts.add(shift)
            return shift
 Duyệt qua từng ảnh
for fname in os.listdir(input_folder):
    input_path = os.path.join(input_folder, fname)
    img = cv2.imread(input_path)
    if img is None:
     1. Khử nhiễu (Median)
    denoised = cv2.medianBlur(img, 5)
     2. Đổi sang HSV
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
     3. Thay đổi Hue ngẫu nhiên không trùng
    shift = get_unique_hue_shift()
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180  # Hue nằm trong [0, 179]
     4. Chuyển lại về BGR
    hsv_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
     5. Lưu kết quả
    output_path = os.path.join(output_folder, f"hsv_{fname}")
    cv2.imwrite(output_path, hsv_shifted)
    print(f"Đã xử lý và lưu: {output_path} (Hue shift: {shift})")
