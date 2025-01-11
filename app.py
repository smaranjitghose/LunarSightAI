import os
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
import moondream as md
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_uploaded_image(uploaded_file):
    """Process uploaded image file and create temporary file."""
    suffix = f".{uploaded_file.type.split('/')[1]}"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getbuffer())
    
    img = Image.open(temp_file.name)
    return img, Path(temp_file.name)

def plot_objects_on_image(image, bounding_boxes, box_color="red", box_width=2):
    """Draw bounding boxes for detected objects."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for bbox in bounding_boxes:
        x_min = int(bbox['x_min'] * width)
        y_min = int(bbox['y_min'] * height)
        x_max = int(bbox['x_max'] * width)
        y_max = int(bbox['y_max'] * height)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=box_width)
    return image

def plot_point_on_image(image, points, point_color="red", point_radius=5):
    """Draw points for detected objects."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for point in points:
        x = int(point['x'] * width)
        y = int(point['y'] * height)
        bounding_box = [
            (x - point_radius, y - point_radius),
            (x + point_radius, y + point_radius)
        ]
        draw.ellipse(bounding_box, fill=point_color)
    return image

def main():
    st.set_page_config(
        page_title="LunarSight AI",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🌙 LunarSight AI")

    # Add API key input in sidebar
    with st.sidebar:
        api_key = st.text_input("Enter your Moondream API Key", type="password")
        if api_key:
            os.environ["MOONDREAM_API_KEY"] = api_key
            
        uploaded_file = st.file_uploader(
            "Upload your image here:",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Initialize Moondream model if API key is provided
    if api_key:
        model = md.vl(api_key=api_key)
    else:
        st.warning("Please enter your Moondream API Key in the sidebar")
        return

    # Create tabs for different tasks
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Image Captioning", 
        "🎯 Object Detection", 
        "📍 Pointing",
        "🔍 Visual Querying"
    ])

    if uploaded_file:
        img, temp_path = process_uploaded_image(uploaded_file)
        
        # Image Captioning Tab
        with tab1:
            st.header("Image Captioning")
            if st.button("Generate Caption"):
                with st.spinner("Generating caption..."):
                    encoded_image = model.encode_image(img)
                    caption = model.caption(encoded_image)["caption"]
                    st.success(f"Caption: {caption}")

        # Object Detection Tab
        with tab2:
            st.header("Object Detection")
            object_query = st.text_input(
                "Enter object to detect:",
                placeholder="Example: person, car, dog"
            )
            if object_query and st.button("Detect Objects"):
                with st.spinner("Detecting objects..."):
                    detect_result = model.detect(img, object_query)
                    if detect_result['objects']:
                        output_img = plot_objects_on_image(img.copy(), detect_result['objects'])
                        st.image(output_img, caption="Detected Objects", use_column_width=True)
                        
                        # Add download button
                        buffered = BytesIO()
                        output_img.save(buffered, format="PNG")
                        st.download_button(
                            "📥 Download Analyzed Image",
                            buffered.getvalue(),
                            "detected_objects.png",
                            "image/png"
                        )
                    else:
                        st.info(f"No {object_query} detected in the image.")

        # Pointing Tab
        with tab3:
            st.header("Object Pointing")
            point_query = st.text_input(
                "Enter object to point at:",
                placeholder="Example: face, hand, logo"
            )
            if point_query and st.button("Point at Object"):
                with st.spinner("Processing..."):
                    point_result = model.point(img, point_query)
                    if point_result["points"]:
                        output_img = plot_point_on_image(img.copy(), point_result["points"])
                        st.image(output_img, caption="Points Detected", use_column_width=True)
                        
                        # Add download button
                        buffered = BytesIO()
                        output_img.save(buffered, format="PNG")
                        st.download_button(
                            "📥 Download Analyzed Image",
                            buffered.getvalue(),
                            "pointed_objects.png",
                            "image/png"
                        )
                    else:
                        st.info(f"No {point_query} detected in the image.")

        # Visual Querying Tab
        with tab4:
            st.header("Visual Querying")
            query = st.text_input(
                "Ask a question about the image:",
                placeholder="Example: What colors are present in this image?"
            )
            if query and st.button("Get Answer"):
                with st.spinner("Analyzing..."):
                    encoded_image = model.encode_image(img)
                    answer = model.query(encoded_image, query)["answer"]
                    st.success(f"Answer: {answer}")

        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            print(f"Could not delete temporary file: {e}")

if __name__ == "__main__":
    main()