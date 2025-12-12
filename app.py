# code by : TranPhuocPhong
import streamlit as st
from PIL import Image
# import sys, os # Có thể xóa luôn cả import sys nếu không dùng nữa

from src.utils import animal_info
from src.inference import transform




uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"]) # tải ảnh lên  Chỉ cho phép các định dạng ảnh phổ biến

if uploaded is not None: # Nếu có ảnh được tải lên
    img = Image.open(uploaded).convert("RGB") # Mở ảnh và chuyển sang chế độ RGB
    st.image(img, caption="Ảnh tải lên", use_column_width=True) # Hiển thị ảnh đã tải lên

    if st.button(" Dự đoán"):
        inf, result = transform(img) # Dự đoán loài động vật và độ tin cậy
        if result < 0.9:
            desc = "_ đây không phải là loài động vật nằm trong diện quý hiếm  ._"
        else:
            desc = animal_info.get(inf)
        st.subheader(" Thông tin loài")
        st.markdown(desc, unsafe_allow_html=True)
