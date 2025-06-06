#import thư viện
import cv2
import numpy as np
import os
# Tạo thư mục đầu vào (exercise) và đầu ra (output_exercise)
input_folder = 'exercise'
output_folder = 'output_exercise'
os.makedirs(output_folder, exist_ok=True)
# Mỗi pixel bị lật ngược màu
def image_inverse(img):
    return 255 - img
#  Làm sáng/tối ảnh tùy theo giá trị gamma
def gamma_correction(img, gamma=2.2):
    img_float = img / 255.0
    corrected = np.power(img_float, 1/gamma)
    corrected_img = np.uint8(corrected * 255)
    return corrected_img
# Làm nổi bật vùng tối trong ảnh 
def log_transformation(img):
    img_float = img.astype(np.float32) + 1
    c = 255 / np.log(1 + np.max(img_float))
    log_img = c * np.log(img_float)
    log_img = np.uint8(np.clip(log_img, 0, 255))
    return log_img
# Cân bằng histogram để ảnh
def histogram_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
# Dãn độ tương phản tuyến tính
def contrast_stretching(img):
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) * (255 / (max_val - min_val))
    stretched = np.uint8(np.clip(stretched, 0, 255))
    return stretched
# Lấy danh sách các file ảnh trong thư mục exercise.
def apply_and_save(transform_func, key_name):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("Không tìm thấy ảnh trong thư mục 'exercise'.")
        return
    # Đọc từng ảnh từ file, bỏ qua nếu ảnh lỗ
    for filename in image_files:
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"Không thể đọc ảnh: {filename}")
            continue
        # Áp dụng hàm biến đổi ảnh, lưu ảnh kết quả vào thư mục output_exercise
        transformed = transform_func(img)
        output_path = os.path.join(output_folder, f"{key_name}_{filename}")
        cv2.imwrite(output_path, transformed)
        print(f"Đã lưu ảnh: {output_path}")
        cv2.imshow(f"Original - {filename}", img)
        cv2.imshow(f"Transformed - {key_name}", transformed)
        key = cv2.waitKey(1000) & 0xFF  # đợi 1000ms
        cv2.destroyAllWindows()
# Hiển thị hướng dẫn người dùng chọn phương pháp biến đổi
print("Chọn phương pháp biến đổi ảnh:")
print("I - Image Inverse")
print("G - Gamma Correction")
print("L - Log Transformation")
print("H - Histogram Equalization")
print("C - Contrast Stretching")
# Nếu lựa chọn hợp lệ, gọi đúng hàm xử lý ảnh. Nếu sai, thông báo lỗi.
key = input("Nhập lựa chọn của bạn (I/G/L/H/C): ").upper()
options = {
    'I': (image_inverse, 'inverse'),
    'G': (gamma_correction, 'gamma'),
    'L': (log_transformation, 'log'),
    'H': (histogram_equalization, 'histogram'),
    'C': (contrast_stretching, 'contrast'),
}

if key in options:
    func, name = options[key]
    apply_and_save(func, name)
else:
    print("Lựa chọn không hợp lệ.")
