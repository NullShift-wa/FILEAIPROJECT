# code by : TranPhuocPhong
import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
train_folder = "D:/AI_PROJECT/assets/processed/train"
val_folder = "D:/AI_PROJECT/assets/processed/val"
if not os.path.exists(train_folder):
    print(f"Không tìm thấy thư mục {train_folder}")
    exit()

if not os.path.exists(val_folder):
    print(f" Không tìm thấy thư mục {val_folder}")
    exit()
# chuẩn bị dữ liệu      
train = ImageDataGenerator(
    rescale=1./255,  # Chuẩn hóa pixel về khoảng [0, 1]
    rotation_range=30, # Xoay ngẫu nhiên trong khoảng 30 độ
    width_shift_range=0.25, # Dịch ngang ngẫu nhiên
    height_shift_range=0.25, # Dịch dọc ngẫu nhiên
    shear_range=0.25, # Cắt xén ngẫu nhiên
    zoom_range=0.3, # Thu phóng ngẫu nhiên
    horizontal_flip=True # Lật ngang ngẫu nhiên
)
# tạo generator cho dữ liệu huấn luyện
train_gen = train.flow_from_directory(
    train_folder,  # Thư mục chứa ảnh huấn luyện
    target_size=(224, 224), # Kích thước ảnh đầu vào
    batch_size=16, # Kích thước batch
    class_mode="categorical" # Phân loại đa lớp với one-hot encoding
)
# tạo generator cho dữ liệu xác thực
val = ImageDataGenerator(
    rescale=1./255 # Chuẩn hóa pixel về khoảng [0, 1]   
    
    )
# tạo generator cho dữ liệu xác thực
val_gen = val.flow_from_directory(
    val_folder, # Thư mục chứa ảnh xác thực
    target_size=(224, 224),  # Kích thước ảnh đầu vào
    batch_size=16, # Kích thước batch
    class_mode="categorical" # Phân loại đa lớp với one-hot encoding
)
# xây dựng mô hình sử dụng MobileNetV2 làm nền tảng
ans = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3)) # Sử dụng trọng số đã được huấn luyện trên ImageNet
ans.trainable = False # Đóng băng các lớp của mô hình nền tảng
# thêm các lớp tùy chỉnh cho bài toán phân loại
x = ans.output # Lấy đầu ra từ mô hình nền tảng
x = GlobalAveragePooling2D()(x) # Thêm lớp pooling toàn cục
x = Dropout(0.2)(x) # Thêm lớp Dropout để giảm overfitting
out = Dense(train_gen.num_classes, activation="softmax")(x) # Lớp đầu ra với hàm kích hoạt softmax cho phân loại đa lớp  
model = Model(inputs=ans.input, outputs=out) # Tạo mô hình hoàn chỉnh
# biên dịch và huấn luyện mô hình  
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) # Biên dịch mô hình với hàm mất mát và tối ưu hóa
model.fit(train_gen, validation_data=val_gen, epochs=20) # Huấn luyện mô hình
model.save("D:\AI_PROJECT\models\model.h5")   
with open("D:\AI_PROJECT\Assets\class_indices.json", "w", encoding="utf-8") as f:  
    json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)
print("done")
    