import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def show_placeholder_image(text, width=200, color="808080"):
    url = f"https://via.placeholder.com/{width}x{width}/{color}/FFFFFF.png?text=" + text.replace(" ","+")
    st.image(url, width=width)

def show_test(test_file, show_gray_placeholders=False):
    c1 = st.columns(1)[0]
    with c1:
        st.markdown("### Test Image")
        if test_file: 
            st.image(Image.open(test_file), width=200)
            gray = Image.open(test_file).convert('L')
            st.image(gray, width=200)
        else: 
            if show_gray_placeholders:
                show_placeholder_image("No Image", color="808080")
            else:
                show_placeholder_image("No Image")

def show_result(test_file, result_img, show_gray_placeholders=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Test Image")
        if test_file: 
            st.image(Image.open(test_file), width=200)
            gray = Image.open(test_file).convert('L')
            st.image(gray, width=200)
        else: 
            if show_gray_placeholders:
                show_placeholder_image("No Image", color="808080")
            else:
                show_placeholder_image("No Image")
    with c2:
        st.markdown("### Closest Match")
        if result_img: 
            st.image(result_img, width=200)
        else: 
            if show_gray_placeholders:
                show_placeholder_image("No Match", color="808080")
            else:
                show_placeholder_image("No Match")

def show_eigenfaces(eigenfaces, num=5, size=100):
    st.markdown("**Eigenfaces Visualization Calculation:**")
    cols = st.columns(num)
    actual_size = int(np.sqrt(eigenfaces.shape[0]))
    for i in range(min(num, eigenfaces.shape[1])):
        v = eigenfaces[:,i]
        v_range = np.ptp(v)
        if v_range > 1e-8:
            img = ((v - v.min()) / v_range * 255).reshape(actual_size, actual_size).astype(np.uint8)
        else:
            img = np.full((actual_size, actual_size), 128, dtype=np.uint8)
        if actual_size != size:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((size, size))
            img = np.array(pil_img)
        with cols[i]:
            st.image(img, caption=f"Eigenface {i+1} ", width=size)

def show_eigenvectors(eigenvectors, num=5):
    st.markdown("**Eigenvector Values Calculation:**")
    eigenvector_data = []
    for i in range(min(num, eigenvectors.shape[0])):
        norm= 0.0
        for val in eigenvectors[i]:
            norm += val * val
        norm = np.sqrt(norm)
        eigenvector_data.append({
            'Eigenvector': f'EV{i+1} ',
            'First 10 Values': str(np.round(eigenvectors[i, :10], 4).tolist()),
            'Length': eigenvectors.shape[1],
            'Norm': f"{norm:.4f}"
        })
    eigenvector_df = pd.DataFrame(eigenvector_data)
    st.dataframe(eigenvector_df, use_container_width=True)

def show_eigenvalue_analysis(eigenvalues, eigenvectors):
    st.markdown("#### Eigenvalue Analysis  Calculation")
    total_eigenvalues = 0.0
    for val in eigenvalues:
        total_eigenvalues += val
    eigenval_data = []
    for i, val in enumerate(eigenvalues):
        contribution = (val / total_eigenvalues * 100) if total_eigenvalues > 0 else 0
        eigenval_data.append({
            'Eigenface': f'EF{i+1} ',
            'Eigenvalue': val,
            'Contribution %': contribution
        })
    eigenval_df = pd.DataFrame(eigenval_data)
    st.dataframe(
        eigenval_df.style.format({
            'Eigenvalue': '{:.2f}',
            'Contribution %': '{:.2f}%'
        }),
        use_container_width=True
    )

def show_weight_comparison(test_weights, training_weights, names, matched_idx):
    st.markdown("#### Weight Comparison Calculation")
    comparison_data = []
    for i in range(len(test_weights)):
        diff = abs(test_weights[i] - training_weights[i, matched_idx])
        comparison_data.append({
            'Eigenface': f'EF{i+1}',
            'Test Image': test_weights[i],
            'Matched Image': training_weights[i, matched_idx],
            ' Difference': diff
        })
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(
        comparison_df.style.format({
            'Test Image': '{:.3f}',
            'Matched Image': '{:.3f}',
            'Difference': '{:.3f}'
        }),
        use_container_width=True
    )
    st.markdown("#### Weight Visualization ")
    chart_data = pd.DataFrame({
        'Test Image': test_weights,
        'Matched Image': training_weights[:, matched_idx]
    }, index=[f'EF{i+1}' for i in range(len(test_weights))])
    st.line_chart(chart_data)

def show_distance_analysis(distance_df, threshold):
    st.markdown("#### Distance Analysis  Euclidean Distance")
    st.write(f"**Current Threshold:** {threshold}")
    def color_distances(val):
        if 'Best Match' in str(val):
            return 'background-color: lightgreen'
        elif 'Within Threshold' in str(val):
            return 'background-color: lightblue'
        else:
            return 'background-color: lightcoral'
    styled_df = distance_df.style.format({'Distance': '{:.3f}'}).applymap(
        color_distances, subset=['Match']
    )
    st.dataframe(styled_df, use_container_width=True)

def show_calculation_details(eigenface_data):
    if 'eigenvalues' in eigenface_data:
        st.markdown("**Sample Calculations:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**First 3 Eigenvalues :**")
            for i, val in enumerate(eigenface_data['eigenvalues'][:3]):
                st.write(f"{i+1} = {val:.6f}")
        with col2:
            st.markdown("** Distance Example:**")
            if 'test_weights' in eigenface_data and 'training_weights' in eigenface_data:
                test_w = eigenface_data['test_weights'][:3]
                train_w = eigenface_data['training_weights'][:3, 0]
                st.write("Test weights: ", [f"{w:.3f}" for w in test_w])
                st.write("Train weights:", [f"{w:.3f}" for w in train_w])
                dist = 0.0
                for i in range(3):
                    diff = test_w[i] - train_w[i]
                    dist += diff * diff
                    st.write(f"(w1[{i}] - w2[{i}])² = ({test_w[i]:.3f} - {train_w[i]:.3f})² = {diff*diff:.6f}")
                dist = np.sqrt(dist)
                st.write(f"** Distance = √{dist*dist:.6f} = {dist:.6f}**")

def show_pca_details(eigvals, var_ratio, projections, names):
    st.markdown("#### Legacy PCA View")
    st.info("This view has been replaced with  eigenface-specific analysis above.")

def show_eigenface_reconstruction(original_img, eigenfaces, mean_face, projection, num_components_list=[1, 3, 5, 10]):
    st.markdown("#### Face Reconstruction ")
    cols = st.columns(len(num_components_list) + 1)
    with cols[0]:
        st.image(original_img, caption="Original", width=80)
    for idx, n_comp in enumerate(num_components_list):
        if n_comp <= eigenfaces.shape[1]:
            reconstructed = mean_face.flatten().copy()
            for i in range(n_comp):
                weight = projection[i]
                eigenface_i = eigenfaces[:, i]
                for j in range(len(reconstructed)):
                    reconstructed[j] += weight * eigenface_i[j]
            reconstructed_img = reconstructed.reshape(50, 50)
            reconstructed_img = ((reconstructed_img - reconstructed_img.min()) / 
                               (reconstructed_img.max() - reconstructed_img.min() + 1e-8) * 255).astype(np.uint8)
            with cols[idx + 1]:
                st.image(reconstructed_img, caption=f"{n_comp} EFs ", width=80)
