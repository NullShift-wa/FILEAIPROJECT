import streamlit as st
from PIL import Image

from src.utils.animal_info import animal_info
from src.inference.predict import transform

uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Ảnh tải lên", use_column_width=True)

    if st.button("Dự đoán"):
        inf, result = transform(img)
        if result < 0.9:
            desc = "_đây không phải là loài động vật nằm trong diện quý hiếm._"
        else:
            desc = animal_info.get(inf)
        st.subheader("Thông tin loài")
        st.markdown(desc, unsafe_allow_html=True)
