
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# モデル読み込み
model = load_model("keras_model.h5")
class_names = [line.strip() for line in open("labels.txt", "r")]

st.set_page_config(page_title="閂AI判定アプリ", layout="centered")

st.title("閂 開閉判定AI Webアプリ")

st.write("""
**撮影方法の注意**
- 閂が**画面中央に大きく映る**ように撮影してください。
- 斜め・遠距離・暗所は誤判定の原因になります。
""")

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

img_source = st.radio(
    "画像の入力方法を選択してください。",
    ("画像をアップロード", "カメラで撮影")
)

if img_source == "画像をアップロード":
    img_file = st.file_uploader("画像を選択してください", type=["png", "jpg", "jpeg"])
else:
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")

    st.subheader("① 元画像（撮影画像）")
    st.image(image, width=480)

    # 正方形化（pad方式）
    image_proc = ImageOps.pad(image, (224,224), color=(255,255,255))

    st.subheader("② AI入力画像（正方形化）")
    st.image(image_proc, width=224)

    image_array = np.asarray(image_proc).astype(np.float32)
    normalized_image_array = (image_array / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    st.subheader("③ 判定結果（円グラフ）")
    fig, ax = plt.subplots()
    ax.pie(prediction[0], labels=class_names, autopct="%.2f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    st.subheader("④ 判定確率（数値）")
    df = pd.DataFrame(prediction[0], index=class_names, columns=["確率"])
    st.dataframe(df)

    max_idx = np.argmax(prediction[0])
    st.success(f"判定結果：**{class_names[max_idx]}**")

