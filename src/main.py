import streamlit as st
from upload import input_dataset_path, upload_test_image
from layout import (show_test, show_eigenfaces, show_placeholder_image, 
                   show_eigenvalue_analysis, show_distance_analysis, 
                   show_eigenvectors, show_result, show_calculation_details)
from recognition import recognize_face

st.set_page_config(page_title="Face Recognition - Calculations", layout="wide")
st.markdown("<h2 style='text-align: center;'>Face Recognition with Eigenface Calculations</h2>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    dataset_path = input_dataset_path()
    test_image_file = upload_test_image()
    threshold = st.slider("Matching Threshold", min_value=50.0, max_value=1000.0, value=300.0, step=10.0)
    st.caption("Lower threshold = stricter matching") 
    with st.expander("ℹ️ Calculation Info"):
        st.markdown("""
        **What's calculated :**
        - Covariance matrix 
        - Eigenvalues 
        - Eigenvectors 
        - Euclidean distance 
        - Matrix multiplication
        - Vector projections 
        """)
    st.markdown("### Recognition Result")
    result_placeholder = st.empty()
    result_placeholder.markdown("<span style='color:green;'>None</span>", unsafe_allow_html=True)

with col2:
    st.markdown("### Eigenface Analysis")
    analysis_placeholder = st.empty()
    
if dataset_path and test_image_file:
    with st.spinner("🔄 Executing face recognition with calculations..."):
        try:
            name, match_img, exec_time, eigenface_data, is_match, distance = recognize_face(
                dataset_path, test_image_file, threshold=threshold, target_size=(50, 50), num_components=5
            )
            with col1:
                if is_match:
                    result_placeholder.markdown(f"<span style='color:green;'>✅ Matched: {name}</span><br>"
                                                f"<span style='color:blue;'> Distance: {distance:.2f}</span>",
                                                unsafe_allow_html=True)
                    show_result(test_image_file, result_img=match_img, show_gray_placeholders=False)
                else:
                    result_placeholder.markdown(f"<span style='color:red;'>❌ No Match Found</span><br>"
                                                f"<span style='color:orange;'>Closest: {name} ( Distance: {distance:.2f})</span>",
                                                unsafe_allow_html=True)
                    show_result(test_image_file, result_img=match_img, show_gray_placeholders=False)
                st.markdown(f"Execution time : <span style='color:green;'>{exec_time:.2f} seconds</span><br>"
                            f"<small style='color:gray;'>✅ calculations completed</small>",
                            unsafe_allow_html=True)
            with col2:
                with analysis_placeholder.container():
                    st.markdown("#### Eigenfaces ")
                    show_eigenfaces(eigenface_data['eigenfaces'], num=5, size=80)
                    st.markdown("#### Mean Face")
                    mean_img = eigenface_data['mean_face'].reshape(50, 50)
                    mean_img_normalized = ((mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8) * 255).astype('uint8')
                    st.image(mean_img_normalized, caption="Average face from dataset", width=100)
                    st.markdown("#### Eigenvectors ")
                    show_eigenvectors(eigenface_data['eigenvectors'], num=5)
                    show_eigenvalue_analysis(eigenface_data['eigenvalues'], eigenface_data['eigenvectors'])
                    show_distance_analysis(eigenface_data['distance_analysis'], threshold)
                    st.markdown("#### Calculation Summary")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Eigenfaces Found", len(eigenface_data['eigenvalues']))
                    with col_b:
                        if len(eigenface_data['eigenvalues']) > 0:
                            dominant_eigenval = eigenface_data['eigenvalues'][0]
                            st.metric("Dominant Eigenvalue", f"{dominant_eigenval:.2f}")
                        else:
                            st.metric("Dominant Eigenvalue", "N/A")
                    with col_c:
                        total_images = eigenface_data['training_weights'].shape[1]
                        st.metric("Training Images", total_images)
        except Exception as e:
            st.error(f"Error during face recognition: {str(e)}")
            with col1:
                result_placeholder.markdown("<span style='color:red;'>❌ Error occurred</span>", unsafe_allow_html=True)
                st.markdown("Execution time : <span style='color:red;'>Error</span>", unsafe_allow_html=True)
else:
    with col2:
        with analysis_placeholder.container():
            st.info("📁 Upload both dataset and test image to see eigenface analysis")
            st.markdown("#### Eigenfaces Preview")
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    show_placeholder_image(f"EF{i+1}", width=80, color="808080")
    with col1:
        st.markdown("Execution time : <span style='color:green;'>00.00 seconds</span>",
                    unsafe_allow_html=True)
