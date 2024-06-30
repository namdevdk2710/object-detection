import cv2
import numpy as np
from PIL import Image
import streamlit as st

MODEL_PATH = "model/MobileNetSSD_deploy.caffemodel"
PROTOTXT_PATH = "model/MobileNetSSD_deploy.prototxt.txt"
CONFIDENCE_THRESHOLD = 0.5


def load_model(prototxt_path, model_path):
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


def process_image(image, net):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    return net.forward()


def annotate_image(image, detections, confidence_threshold):
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(image, (start_x, start_y),
                          (end_x, end_y), (255, 0, 0), 2)
    return image


def main():
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

    if file is not None:
        st.image(file, caption="Uploaded Image")

        image = Image.open(file)
        image = np.array(image)

        net = load_model(PROTOTXT_PATH, MODEL_PATH)
        detections = process_image(image, net)
        processed_image = annotate_image(
            image, detections, CONFIDENCE_THRESHOLD)

        st.image(processed_image, caption="Processed Image")


if __name__ == "__main__":
    main()
