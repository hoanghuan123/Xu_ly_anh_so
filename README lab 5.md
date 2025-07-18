# 1
Load ảnh và chuyển sang ảnh xám
python
Sao chép
Chỉnh sửa
data = Image.open('geometric.png').convert('L')
a = np.asarray(data)
Mở ảnh "geometric.png" và chuyển thành ảnh xám ('L' = 8-bit grayscale).

Chuyển ảnh thành mảng NumPy để xử lý.

2. Áp dụng ngưỡng Otsu để phân ngưỡng ảnh
python
Sao chép
Chỉnh sửa
thres = threshold_otsu(a)
b = a > thres
Tính ngưỡng tối ưu bằng thuật toán Otsu.

Tạo ảnh nhị phân (b) với các pixel sáng hơn ngưỡng sẽ là True (1), còn lại là False (0).

3. Gán nhãn các vùng liên thông
python
Sao chép
Chỉnh sửa
c = label(b)
Dùng skimage.morphology.label để gán số hiệu (label) cho từng vùng liên thông (connected components).

4. Lưu ảnh đã gán nhãn (tùy chọn)
python
Sao chép
Chỉnh sửa
cl = Image.fromarray((c * (255 / np.max(c))).astype(np.uint8))
iio.imsave('label_output.jpg', cl)
Chuyển ảnh nhãn c thành ảnh hiển thị được bằng cách chuẩn hóa về mức xám 0-255.

Lưu ảnh kết quả để kiểm tra trực quan.

5. Trích xuất thuộc tính của từng vùng
python
Sao chép
Chỉnh sửa
regions = regionprops(c)
Sử dụng regionprops để lấy thông tin về từng vùng: bounding box, centroid, diện tích, v.v.

6. Vẽ ảnh đã gán nhãn
python
Sao chép
Chỉnh sửa
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(c, cmap='nipy_spectral')
Tạo khung vẽ và hiển thị ảnh nhãn c bằng màu giả (nipy_spectral) để phân biệt các vùng.

7. Vẽ bounding box và centroid cho từng đối tượng
python
Sao chép
Chỉnh sửa
for i, region in enumerate(regions):
    minr, minc, maxr, maxc = region.bbox
    ...
    ax.add_patch(rect)
    ...
    ax.plot(cx, cy, 'ro')
    ax.text(cx, cy, str(i + 1), ...)
Với mỗi vùng:

Vẽ hộp giới hạn (bounding box) bao quanh đối tượng.

Đánh dấu tâm đối tượng (centroid) bằng dấu chấm đỏ.

Đánh số thứ tự từng đối tượng tại vị trí tâm.

8. Tùy chỉnh và hiển thị kết quả
python
Sao chép
Chỉnh sửa
ax.set_title('Labeled Objects with Bounding Boxes and Centroids')
ax.axis('off')
plt.tight_layout()
plt.show()
Tắt trục tọa độ, thêm tiêu đề và hiển thị ảnh.


# 2

Import các thư viện để xử lý ảnh, toán học và vẽ biểu đồ.

python
Sao chép
Chỉnh sửa
data = Image.open('geometric.png').convert('L')
a = np.asarray(data)
bmg = abs(a - nd.shift(a, (0, 1), order=0))
Image.open(...).convert('L'): đọc ảnh và chuyển sang grayscale.

nd.shift(a, (0, 1)): dịch ảnh sang phải 1 pixel.

abs(...): tính chênh lệch giữa ảnh gốc và ảnh dịch → phát hiện biên theo chiều ngang.

# 3

Harris Corner Detection - Phát hiện góc bằng phương pháp Harris

## Mục tiêu
Áp dụng thuật toán **Harris Corner Detection** để phát hiện các góc (corner points) trong ảnh xám. Đây là kỹ thuật quan trọng trong thị giác máy tính để nhận dạng đặc trưng.

---

## Thư viện sử dụng

```python
from PIL import Image
import numpy as np
import cv2
import imageio.v2 as iio
import scipy.ndimage as nd
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from skimage.filters.thresholding import threshold_otsu
 Giải thích thuật toán Harris
python
Sao chép
Chỉnh sửa
def Harris(indata, alpha=0.2):
    x = nd.sobel(indata, 0)     # I_x: đạo hàm theo trục x
    y = nd.sobel(indata, 1)     # I_y: đạo hàm theo trục y

    x1 = x ** 2                 # I_x^2
    y1 = y ** 2                 # I_y^2
    xy = abs(x * y)            # |I_x * I_y|

    # Làm mượt bằng Gaussian filter
    x1 = nd.gaussian_filter(x1, 3)
    y1 = nd.gaussian_filter(y1, 3)
    xy = nd.gaussian_filter(xy, 3)

    # Ma trận đặc trưng Harris: R = det(C) - alpha * (trace(C))^2
    detC = x1 * y1 - 2 * xy
    trC = x1 + y1
    R = detC - alpha * trC**2

    return R
Hàm Harris() trả về một ảnh R chứa giá trị mức độ "góc" tại mỗi điểm ảnh.
Các giá trị lớn trong R là nơi có khả năng cao là góc (corner).

# 4
Các bước thực hiện
1. Đọc ảnh và chuyển sang grayscale
python
Sao chép
Chỉnh sửa
data = iio.imread('bird.png')
image_gray = rgb2gray(data)  # Chuyển ảnh màu sang ảnh xám
rgb2gray() giúp giảm chiều dữ liệu từ ảnh màu (3 kênh) thành ảnh đơn kênh (1 kênh), thuận lợi cho xử lý góc.

2. Tính ảnh phản hồi góc (corner response)
python
Sao chép
Chỉnh sửa
coordinate = corner_harris(image_gray, k=0.001)
Hàm corner_harris trả về một ma trận cường độ góc, chứa giá trị cao ở các điểm có khả năng là góc ảnh.

Tham số k điều chỉnh độ nhạy với góc. Giá trị nhỏ như 0.001 cho phép phát hiện cả những góc nhỏ.

3. Hiển thị ảnh phản hồi góc
python
Sao chép
Chỉnh sửa
plt.figure(figsize=(20, 10))
plt.imshow(coordinate)
plt.axis('off')
plt.show()
Các điểm sáng thể hiện vị trí có khả năng là góc.

Có thể tiếp tục xử lý để lọc góc nổi bật hoặc đánh dấu trên ảnh gốc (xem phần mở rộng bên dưới).

Mở rộng: Đánh dấu góc rõ nhất trên ảnh
Anh có thể sử dụng hàm corner_peaks để lấy tọa độ cụ thể các điểm góc, rồi vẽ chấm tròn lên ảnh:

python
Sao chép
Chỉnh sửa
from skimage.feature import corner_peaks

coords = corner_peaks(coordinate, min_distance=5)
plt.imshow(data)
plt.scatter(coords[:, 1], coords[:, 0], s=40, c='red', marker='o')
plt.title('Detected Corners on Image')
plt.axis('off')
plt.show()

