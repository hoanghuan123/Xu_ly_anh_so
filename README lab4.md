 1. Đọc ảnh và tiền xử lý

```python
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
import imageio.v2 as iio

image = iio.imread('bird.png')
gray = rgb2gray(image)
Ảnh đầu vào có thể là ảnh màu (bird.png)

Chuyển đổi ảnh về dạng grayscale để xử lý

2. Phát hiện góc bằng thuật toán Harris
Cách 1: Tự cài đặt Harris Corner (thuật toán thủ công)
python
Sao chép
Chỉnh sửa
def Harris(indata, alpha=0.2):
    # Tính đạo hàm theo x và y
    ...
    # Tính ma trận đặc trưng R
    R = detC - alpha * trC**2
    return R
Hàm Harris() trả về ma trận phản hồi góc, giá trị cao thể hiện vùng có khả năng là góc.

Cách 2: Sử dụng corner_harris từ thư viện skimage
python
Sao chép
Chỉnh sửa
from skimage.feature import corner_harris

R = corner_harris(gray, k=0.001)
3. Hiển thị kết quả
python
Sao chép
Chỉnh sửa
plt.imshow(R, cmap='gray')
plt.title("Corner Response")
plt.axis('off')
plt.show()
Hiển thị ảnh phản hồi góc, nơi giá trị lớn tương ứng với các điểm góc.

 4. Đánh dấu các điểm góc (mở rộng)
python
Sao chép
Chỉnh sửa
from skimage.feature import corner_peaks

coords = corner_peaks(R, min_distance=5)
plt.imshow(image)
plt.scatter(coords[:, 1], coords[:, 0], c='red', s=20)
plt.title("Detected Corners")
plt.axis('off')
plt.show()
Đánh dấu trực tiếp các điểm góc trên ảnh gốc để trực quan hóa kết quả.
