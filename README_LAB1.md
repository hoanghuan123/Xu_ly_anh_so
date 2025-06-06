 # Câu 1: 
 #1: Import thư viện  
 32: Đọc ảnh đầu vào  : img = imageio.imread('flower.jpeg')  
 #3: Tách từng kênh màu : r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]  
 #4: Tạo ảnh chỉ có một kênh màu :   
img_r = np.stack((r, np.zeros_like(r), np.zeros_like(r)), axis=2)
img_g = np.stack((np.zeros_like(g), g, np.zeros_like(g)), axis=2)
img_b = np.stack((np.zeros_like(b), np.zeros_like(b), b), axis=2)
 #5: Ghi các ảnh kết quả ra file:
imageio.imwrite('red.png', img_r.astype(np.uint8))
imageio.imwrite('green.png', img_g.astype(np.uint8))
imageio.imwrite('blue.png', img_b.astype(np.uint8))
# Kết quả:
![blue](https://github.com/user-attachments/assets/bd7409a9-509f-4bda-9a35-3d17197e98ef)
![red](https://github.com/user-attachments/assets/4eba5a9c-f7e3-4897-83c8-83467bb26d7c)
![green](https://github.com/user-attachments/assets/f60354e4-d323-4271-ac9c-14df81a899cf)

# Câu 2: 
# 1: Đọc ảnh gốc :img = imageio.imread('balloons_noisy.png')
# 2: Tạo ảnh với kênh màu hoán đổi :
img_bgr = img[:, :, ::-1]
img_grb = img[:, :, [1, 0, 2]]
img_brg = img[:, :, [2, 0, 1]]
3: Lưu các ảnh kết quả ra file:
imageio.imwrite('bgr.jpg', img_bgr)
imageio.imwrite('grb.jpg', img_grb)
imageio.imwrite('brg.jpg', img_brg)
# Kết quả
![brg](https://github.com/user-attachments/assets/cbcdd7b8-bc3f-42ff-85fe-54ac81ba2a2b)
![grb](https://github.com/user-attachments/assets/0824a12a-9d58-43d4-afb7-99b34f5c5ad1)
![bgr](https://github.com/user-attachments/assets/acfc6be6-ac71-4407-b05c-a9d3ecd5598c)
# câu 3:

# Import thư viện OpenCV
# 1: Đọc ảnh gốc : img = cv2.imread('bird.png')
# 2: Chuyển sang hệ màu HSV : hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 3: Tách 3 kênh HSV :h, s, v = cv2.split(hsv)
# 4: Ghi ảnh ra file:
cv2.imwrite('h.jpg', h)
cv2.imwrite('s.jpg', s)
cv2.imwrite('v.jpg', v)
 # Kết quả: 
h.jpg: hiển thị vùng tông màu.
s.jpg: hiển thị độ đậm nhạt của màu.
v.jpg: hiển thị độ sáng.
kết quả:
![h](https://github.com/user-attachments/assets/35b5facb-3bfe-4b68-864a-c1f9ed5b05cf)
![s](https://github.com/user-attachments/assets/ccd758d0-1c5d-42e5-88d4-e52f2f9853dc)
![v](https://github.com/user-attachments/assets/84de7c26-9537-454e-bde2-7d680d7c77f8)

# câu 4:
1: Đọc ảnh và chuyển từ BGR sang HSV:
img = cv2.imread('balloons_noisy.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
2: Biến đổi kênh Hue và Value:
hsv[:, :, 0] = hsv[:, :, 0] / 3
hsv[:, :, 2] = hsv[:, :, 2] * 0.75
3: Giới hạn giá trị hợp lệ của H và V:
hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
4: Chuyển lại HSV → BGR và lưu ảnh:
hsv = hsv.astype(np.uint8)
result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('hsv.png', result)
plt.imshow(result)
plt.axis('off')
plt.show()
 # Kết quả
![hsv](https://github.com/user-attachments/assets/690d420d-6191-4a23-8f62-5f3f99a165b1)

# câu 5: 
# Tạo thư mục lưu kết quả (mean_filter):
  input_folder = 'exercise'
  output_folder = 'mean_filter'
  os.makedirs(output_folder, exist_ok=True)
# Bộ lọc trung bình 5x5:
  k = np.ones((5, 5)) / 25
# Duyệt qua từng file ảnh trong thư mục
  for filename in os.listdir(input_folder):
  if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
  input_path = os.path.join(input_folder, filename)
  output_path = os.path.join(output_folder, filename)

# Đọc ảnh grayscale
  a = iio.imread(input_path,mode ='F')
# Áp dụng bộ lọc trung bình
  b = sn.convolve(a, k).astype(np.uint8)
# Lưu ảnh sau khi lọc
  iio.imsave(output_path, b)
# Hiển thị ảnh kết quả (tuỳ chọn)
  plt.imshow(b, cmap='gray')
  plt.axis('off')
  plt.show()


![baby](https://github.com/user-attachments/assets/0cb27afe-426c-4c02-b5df-35678ea61d8e)
![flower](https://github.com/user-attachments/assets/5d141385-a6ed-47d3-aa2b-3ba80c00b596)
![balloons_noisy](https://github.com/user-attachments/assets/2fb8b325-4db3-46e3-95ec-5279f58c1ebd)



