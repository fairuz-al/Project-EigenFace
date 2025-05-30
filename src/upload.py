import streamlit as st
import os

def input_dataset_path():
    st.markdown("### Dataset Folder Path")
    folder_path = st.text_input(
        "Enter the path to your dataset folder:",
        placeholder="e.g., /path/to/dataset or C:\\dataset",
        key="dataset_path"
    )
    
    if folder_path and os.path.exists(folder_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(folder_path) if f.endswith(ext)])
        
        if image_files:
            st.success(f"✅ Found {len(image_files)} images in the folder")
            with st.expander("Show image files"):
                for img in image_files[:10]:  
                    st.text(img)
                if len(image_files) > 10:
                    st.text(f"... and {len(image_files) - 10} more")
            return folder_path
        else:
            st.error("❌ No image files found in the specified folder")
            return None
    elif folder_path:
        st.error("❌ Folder path does not exist")
        return None
    else:
        return None

def upload_test_image():
    st.markdown("### Insert Your Test Image")
    return st.file_uploader(
        "Choose File",
        key="test_image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )