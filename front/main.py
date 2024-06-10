import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import json


def draw_boxes(image, boxes, class_names):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xyxy = box['xyxy'][0]
        conf = box['conf']
        cls_id = box['cls']
        label = f"{class_names[str(cls_id)]} {conf:.2f}"
        draw.rectangle(xyxy, outline="red", width=2)
        draw.text((xyxy[0], xyxy[1] - 10), label, fill="red")
    return image


def main():
    st.title("Toxic Text Classification")
    input_text = st.text_area('Enter the text to classify')

    if st.button("Classify!") and input_text:
        # st.write(f'Input text: {input_text}')
        data = {'text': input_text}  # Заменяем 'class_pred' на 'text'
        try:          
            response = requests.post("http://127.0.0.1:8000/clf_text", json=data)
            if response.status_code == 200:               
                res_data = response.json()

                st.write(f"Class: {res_data['class_pred']}")
                st.write(f"Prediction score: {res_data['probability']}")
            else:
                st.error(f"Error: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")


    st.title("Image detecting")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()

        if st.button("Detect Objects"):
            response = requests.post("http://127.0.0.1:8000/detect", files={"file": ("filename", img_bytes, "image/jpeg")})
            detections = response.json()['boxes']
            class_names = response.json()['names']
            # st.write(f"Received detections: {detections}")
            # st.write(f"Class names: {class_names}")
            image_with_boxes = draw_boxes(image, detections, class_names)
            st.image(image_with_boxes, caption='Detected Image', use_column_width=True)

if __name__ == '__main__':
    main()