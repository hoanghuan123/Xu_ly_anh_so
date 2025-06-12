#Câu 1:
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


#câu 2
#Biến ảnh xám thành phổ tần số, chủ yếu để trực quan hóa ảnh trong miền tần số.
def fast_fourier_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # chuyển sang ảnh xám
    f = np.fft.fft2(gray)                         # biến đổi Fourier 2D
    fshift = np.fft.fftshift(f)                   # dịch tâm phổ tần số
    spectrum = 20 * np.log(np.abs(fshift) + 1)    # log để hiển thị phổ
    return np.uint8(np.clip(spectrum, 0, 255))    # chuyển về ảnh hiển thị được

if key in options:
    func, name = options[key]
    apply_and_save(func, name)
else:
    print("Lựa chọn không hợp lệ.")

#Tạo hàm Butterworth filter với dạng:
def butterworth_filter(shape, cutoff, order, highpass=False):
    rows, cols = shape
    u = np.arange(rows) - rows//2
    v = np.arange(cols) - cols//2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    if highpass:
        H = 1 / (1 + (cutoff / (D + 1e-5)) ** (2 * order))
    else:
        H = 1 / (1 + (D / cutoff) ** (2 * order))
    return H
# Biến đổi ảnh sang miền tần số, nhân với mặt nạ Butterworth, rồi biến đổi ngược lại về miền không gian
def apply_butterworth(img, cutoff, order, highpass):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    H = butterworth_filter(gray.shape, cutoff, order, highpass)
    filtered = fshift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.uint8(np.abs(img_back))
#Gán hàm biến đổi đúng theo lựa chọn người dùng.
key = input("Chọn F (FFT), L (Lowpass), H (Highpass): ").upper()

if key == 'F':
    func = fast_fourier_transform
elif key == 'L':
    func = lambda img: apply_butterworth(img, cutoff=30, order=2, highpass=False)
elif key == 'H':
    func = lambda img: apply_butterworth(img, cutoff=30, order=2, highpass=True)
else:
    print("Phím không hợp lệ.")
    exit()
# Với mỗi ảnh:

Đọc ảnh gốc

Áp dụng biến đổi

Hiển thị kết quả

Lưu vào thư mục output_exercise/
for filename in os.listdir("exercise"):
    path = os.path.join("exercise", filename)
    img = cv2.imread(path)
    if img is not None:
        output = func(img)
        cv2.imwrite(f"output_exercise/{key}_{filename}", output)
        cv2.imshow("Original", img)
        cv2.imshow("Transformed", output)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

# Câu 3
import cv2
import numpy as np
import os
import random

# Thư mục chứa ảnh gốc và thư mục để lưu ảnh sau khi xử lý
input_folder = 'exercise'
output_folder = 'output_exercise'
os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục nếu chưa có

# ==== Các hàm biến đổi ảnh từ câu 1 ====

# Đảo ngược màu ảnh (âm bản)
def image_inverse(img):
    return 255 - img

# Biến đổi gamma (hiệu chỉnh độ sáng gamma)
def gamma_correction(img, gamma=2.2):
    img_float = img / 255.0
    corrected = np.power(img_float, 1/gamma)
    return np.uint8(corrected * 255)

# Biến đổi log (làm rõ chi tiết vùng tối)
def log_transformation(img):
    img_float = img.astype(np.float32) + 1  # tránh log(0)
    c = 255 / np.log(1 + np.max(img_float))  # hệ số scale
    log_img = c * np.log(img_float)
    return np.uint8(np.clip(log_img, 0, 255))

# Cân bằng histogram (cải thiện độ tương phản)
def histogram_equalization(img):
    if len(img.shape) == 2:  # ảnh grayscale
        return cv2.equalizeHist(img)
    else:
        # Chuyển ảnh màu sang YCrCb, cân bằng histogram kênh Y (độ sáng)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Giãn độ tương phản (trải đều giá trị pixel từ min đến max)
def contrast_stretching(img):
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) * (255 / (max_val - min_val))
    return np.uint8(np.clip(stretched, 0, 255))
    # Duyệt qua tất cả ảnh trong thư mục 'exercise'
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Không đọc được ảnh: {file}")
                continue
     # Đổi thứ tự kênh màu từ BGR → RGB bằng cách đảo kênh
            img_rgb = img[:, :, ::-1]
     # Chọn ngẫu nhiên một hàm xử lý trong câu 1
            func = random.choice(funcs)
     # Áp dụng hàm xử lý ảnh đã chọn
            result = func(img_rgb)
     # Lưu ảnh kết quả vào thư mục output
            output_path = os.path.join(output_folder, f'q3_{file}')
            cv2.imwrite(output_path, result)
            print("Đã lưu:", output_path)
# Hiển thị ảnh (tùy chọn – anh có thể bật/tắt)
    
#câu 4
import cv2
import numpy as np
import os
import random

# Thư mục chứa ảnh gốc và thư mục lưu ảnh sau xử lý
input_folder = 'exercise'
output_folder = 'output_exercise'
os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục output nếu chưa có

# ==== Các hàm xử lý từ Câu 2 ====

# Biến đổi Fourier để hiển thị phổ tần số của ảnh xám
def fast_fourier(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang xám
    f = np.fft.fft2(gray)                         # Biến đổi Fourier 2 chiều
    fshift = np.fft.fftshift(f)                   # Dịch tâm phổ
    magnitude = 20 * np.log(np.abs(fshift) + 1)   # Tính biên độ phổ (log scale)
    return np.uint8(np.clip(magnitude, 0, 255))   # Chuyển về ảnh 8-bit hiển thị được

# Lọc thông thấp Butterworth
def butterworth_lowpass(img, D0=30, n=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    # Tạo mặt nạ lọc thông thấp
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(u**2 + v**2)
    H = 1 / (1 + (D / D0)**(2 * n))
    filtered = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(np.clip(img_back, 0, 255))
# Lọc thông cao Butterworth
def butterworth_highpass(img, D0=30, n=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    # Tạo mặt nạ lọc thông cao
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(u**2 + v**2)
    H = 1 - 1 / (1 + (D / D0)**(2 * n))  # 1 - Lowpass = Highpass
    filtered = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(np.clip(img_back, 0, 255))
# Bộ lọc Min – làm mờ vùng tối (lọc cực tiểu)
def min_filter(img):
    return cv2.erode(img, np.ones((3,3), np.uint8))

# Bộ lọc Max – làm sáng vùng sáng (lọc cực đại)
def max_filter(img):
    return cv2.dilate(img, np.ones((3,3), np.uint8))

# ==== Hàm xử lý Câu 4 ====
def question4():
    # Danh sách các hàm biến đổi ảnh từ Câu 2 và tên tương ứng
    functions = [
        (fast_fourier, 'fourier'),
        (butterworth_lowpass, 'lowpass'),
        (butterworth_highpass, 'highpass'),
    ]
    # Duyệt từng ảnh trong thư mục input
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Lỗi đọc ảnh: {file}")
             continue
            # Bước 1: Đổi thứ tự kênh màu BGR → RGB
            img_rgb = img[:, :, ::-1]
            # Bước 2: Chọn ngẫu nhiên 1 trong 3 phép biến đổi ảnh
            func, name = random.choice(functions)
            processed = func(img_rgb)
            # Bước 3: Nếu là Butterworth Lowpass thì lọc thêm bằng Min filter
            if name == 'lowpass':
                processed = min_filter(processed)
        # Nếu là Butterworth Highpass thì lọc thêm bằng Max filter
            elif name == 'highpass':
                processed = max_filter(processed)
            # Bước 4: Lưu ảnh đã xử lý với tên có tiền tố "q4_"
            out_path = os.path.join(output_folder, f"q4_{name}_{file}")
            cv2.imwrite(out_path, processed)
            print("Đã lưu:", out_path)
# ==== Gọi hàm chạy bài Câu 4 ====
question4()

