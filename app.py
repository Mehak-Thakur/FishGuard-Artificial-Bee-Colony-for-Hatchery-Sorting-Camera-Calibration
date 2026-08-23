import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(
    page_title="FishGuard AI",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🐟 FishGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Fish Detection using YOLO11m + Artificial Bee Colony Camera Calibration</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    **FishGuard — Artificial Bee Colony for Hatchery Sorting-Camera Calibration**

    Upload an image or capture an image using your camera to detect and classify fish automatically.
    """
)


MODEL_PATH = os.path.join(
    "models",
    "best.pt"
)


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model not found: `{MODEL_PATH}`"
    )

    st.info(
        """
        Make sure your project has this structure:

        FishGuard/
        ├── app.py
        ├── requirements.txt
        └── models/
            └── best.pt
        """
    )

    st.stop()


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()


st.sidebar.header("⚙️ FishGuard Settings")


confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)


iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)


image_size = st.sidebar.selectbox(
    "Inference Image Size",
    [320, 416, 512, 640, 768],
    index=3
)


st.sidebar.markdown("---")

st.sidebar.subheader(
    "🐝 ABC Camera Calibration"
)

st.sidebar.caption(
    "Optimized parameters obtained from the FishGuard Artificial Bee Colony calibration stage."
)


brightness = st.sidebar.slider(
    "Brightness",
    -100.0,
    100.0,
    25.72657,
    0.1
)


contrast = st.sidebar.slider(
    "Contrast",
    0.5,
    2.0,
    1.00721,
    0.01
)


gamma = st.sidebar.slider(
    "Gamma",
    0.5,
    2.5,
    1.30102,
    0.01
)


saturation = st.sidebar.slider(
    "Saturation",
    0.5,
    2.0,
    1.18816,
    0.01
)


clahe_clip = st.sidebar.slider(
    "CLAHE Clip Limit",
    0.5,
    5.0,
    3.90249,
    0.1
)


use_abc = st.sidebar.checkbox(
    "Enable ABC Calibration",
    value=True
)


def apply_abc_calibration(
    image,
    brightness_value,
    contrast_value,
    gamma_value,
    saturation_value,
    clahe_clip_value
):

    img = image.astype(np.float32)

    img = (
        img * contrast_value
        + brightness_value
    )

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    gamma_value = max(
        gamma_value,
        0.01
    )

    inv_gamma = 1.0 / gamma_value

    table = np.array(
        [
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]
    ).astype(np.uint8)

    img = cv2.LUT(
        img,
        table
    )

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    ).astype(np.float32)

    hsv[:, :, 1] *= saturation_value

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        255
    )

    img = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2RGB
    )

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_value,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        (
            l_channel,
            a_channel,
            b_channel
        )
    )

    img = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return img


st.markdown("---")


input_method = st.radio(
    "Choose Input Method",
    ["Upload Image", "Camera"],
    horizontal=True
)


uploaded_file = None


if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a fish image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="image_uploader"
    )

else:

    uploaded_file = st.camera_input(
        "Capture fish image",
        key="camera_input"
    )


if uploaded_file is not None:

    try:

        pil_image = Image.open(
            uploaded_file
        ).convert("RGB")

        original_image = np.array(
            pil_image
        )

    except Exception as e:

        st.error(
            f"Unable to read image: {e}"
        )

        st.stop()


    if use_abc:

        processed_image = apply_abc_calibration(
            original_image,
            brightness,
            contrast,
            gamma,
            saturation,
            clahe_clip
        )

    else:

        processed_image = original_image.copy()


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Original Image"
        )

        st.image(
            original_image,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "ABC-Calibrated Image"
        )

        st.image(
            processed_image,
            use_container_width=True
        )


    with st.spinner(
        "🐟 Detecting fish..."
    ):

        results = model.predict(
            source=processed_image,
            conf=confidence,
            iou=iou_threshold,
            imgsz=image_size,
            verbose=False
        )


    result = results[0]

    boxes = result.boxes

    if boxes is not None:

        detection_count = len(boxes)

    else:

        detection_count = 0


    annotated_image = result.plot()

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )


    st.markdown("---")

    st.subheader(
        "📊 Detection Results"
    )


    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "🐟 Fish Detected",
            detection_count
        )


    with metric2:

        if detection_count > 0:

            confidences = (
                boxes.conf.cpu().numpy()
            )

            avg_confidence = float(
                np.mean(confidences)
            )

            st.metric(
                "Average Confidence",
                f"{avg_confidence * 100:.2f}%"
            )

        else:

            st.metric(
                "Average Confidence",
                "N/A"
            )


    with metric3:

        if detection_count > 0:

            max_confidence = float(
                boxes.conf.max().item()
            )

            st.metric(
                "Highest Confidence",
                f"{max_confidence * 100:.2f}%"
            )

        else:

            st.metric(
                "Highest Confidence",
                "N/A"
            )


    st.subheader(
        "🎯 Fish Detection"
    )


    st.image(
        annotated_image,
        use_container_width=True
    )


    st.subheader(
        "🏷️ Detected Fish Classes"
    )


    class_counts = {}


    if detection_count > 0:

        class_ids = (
            boxes.cls.cpu().numpy().astype(int)
        )

        confidences = (
            boxes.conf.cpu().numpy()
        )


        for class_id, conf in zip(
            class_ids,
            confidences
        ):

            class_name = model.names[
                int(class_id)
            ]

            if class_name not in class_counts:

                class_counts[class_name] = {
                    "count": 0,
                    "confidences": []
                }

            class_counts[class_name]["count"] += 1

            class_counts[class_name]["confidences"].append(
                float(conf)
            )


    if class_counts:

        rows = []

        for class_name, data in class_counts.items():

            rows.append(
                {
                    "Fish Class": class_name,
                    "Count": data["count"],
                    "Average Confidence":
                        f"{np.mean(data['confidences']) * 100:.2f}%"
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No fish detected above the selected confidence threshold."
        )


    if use_abc:

        with st.expander(
            "🐝 View ABC Calibration Parameters"
        ):

            st.write(
                {
                    "Brightness": brightness,
                    "Contrast": contrast,
                    "Gamma": gamma,
                    "Saturation": saturation,
                    "CLAHE Clip": clahe_clip
                }
            )


    output_bgr = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_RGB2BGR
    )


    success, encoded_image = cv2.imencode(
        ".jpg",
        output_bgr
    )


    if success:

        st.download_button(
            label="⬇️ Download Detection Result",
            data=encoded_image.tobytes(),
            file_name="FishGuard_detection.jpg",
            mime="image/jpeg"
        )


else:

    st.info(
        "👆 Upload a fish image or capture one using your camera to start detection."
    )


st.markdown("---")

st.caption(
    "FishGuard — Artificial Bee Colony for Hatchery Sorting-Camera Calibration | YOLO11m"
)