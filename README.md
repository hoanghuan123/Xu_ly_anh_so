1. Load ảnh và chuyển sang ảnh xám
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







