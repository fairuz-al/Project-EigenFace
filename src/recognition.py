import os
import time
import glob
import numpy as np
import cv2
import pandas as pd
from PIL import Image

def load_images_from_folder(folder_path, target_size=(50, 50)):
    arrs, names = [], []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for ext in extensions:
        for img_path in glob.glob(os.path.join(folder_path, ext)):
            try:
                pil_img = Image.open(img_path)
                cv_img = np.array(pil_img.convert('RGB'))
                gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) > 0:
                    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
                    face_img = pil_img.crop((x, y, x+w, y+h))
                else:
                    face_img = pil_img
                gray_img = face_img.convert('L').resize(target_size)
                arrs.append(np.asarray(gray_img, dtype=np.float32).flatten())
                names.append(os.path.basename(img_path))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    if not arrs:
        raise ValueError(f"No valid images found in folder: {folder_path}")
    X = np.stack(arrs, axis=1)
    return X, names

def covariance_matrix(X_centered):
    n_samples = X_centered.shape[1]
    L = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            dot_product = np.dot(X_centered[:, i], X_centered[:, j])
            L[i, j] = L[j, i] = dot_product / (n_samples - 1)
    return L

def power_iteration(matrix, num_iterations=100, tolerance=1e-6):
    n = matrix.shape[0]
    vector = np.random.rand(n)
    norm = np.sqrt(np.sum(vector * vector))
    vector = vector / norm
    eigenvalue = 0
    prev_eigenvalue = 0
    for i in range(num_iterations):
        new_vector = np.zeros(n)
        for j in range(n):
            for k in range(n):
                new_vector[j] += matrix[j, k] * vector[k]
        eigenvalue = np.dot(vector, new_vector) / np.dot(vector, vector)
        norm = np.sqrt(np.sum(new_vector * new_vector))
        if norm > 1e-12:
            vector = new_vector / norm
        if i > 0 and abs(eigenvalue - prev_eigenvalue) < tolerance:
            break
        prev_eigenvalue = eigenvalue
    return eigenvalue, vector

def eigenvalue_decomposition(matrix, num_components=5):
    eigenvalues = []
    eigenvectors = []
    working_matrix = matrix.copy()
    for i in range(min(num_components, matrix.shape[0])):
        try:
            eigenval, eigenvec = power_iteration(working_matrix)
            if abs(eigenval) < 1e-8:
                break
            eigenvalues.append(eigenval)
            eigenvectors.append(eigenvec)
            outer_product = np.outer(eigenvec, eigenvec) * eigenval
            working_matrix = working_matrix - outer_product
        except Exception as e:
            break
    return np.array(eigenvalues), np.array(eigenvectors)

def compute_eigenfaces(X_centered, num_components=5):
    n_pixels, n_samples = X_centered.shape
    L = covariance_matrix(X_centered)
    eigenvalues, eigenvectors_L = eigenvalue_decomposition(L, num_components)
    eigenfaces = np.zeros((n_pixels, len(eigenvalues)))
    for i in range(len(eigenvalues)):
        eigenvec_full = np.zeros(n_pixels)
        for j in range(n_pixels):
            for k in range(n_samples):
                eigenvec_full[j] += X_centered[j, k] * eigenvectors_L[i, k]
        norm = np.sqrt(np.sum(eigenvec_full * eigenvec_full))
        if norm > 1e-12:
            eigenvec_full = eigenvec_full / norm
        eigenfaces[:, i] = eigenvec_full
    return eigenfaces, eigenvalues, eigenvectors_L

def euclidean_distance(weights1, weights2):
    diff = weights1 - weights2
    return np.sqrt(np.sum(diff * diff))

def recognize_face(folder_path, test_img_file, threshold=300.0, target_size=(50, 50), num_components=5):
    start = time.time()
    X_raw, names = load_images_from_folder(folder_path, target_size)
    mean_face = X_raw.mean(axis=1, keepdims=True)
    X_centered = X_raw - mean_face
    eigenfaces, eigenvalues, eigenvectors = compute_eigenfaces(X_centered, num_components)
    training_weights = np.zeros((eigenfaces.shape[1], X_raw.shape[1]))
    for i in range(X_raw.shape[1]):
        centered_face = X_raw[:, i] - mean_face.flatten()
        for j in range(eigenfaces.shape[1]):
            training_weights[j, i] = np.dot(eigenfaces[:, j], centered_face)
    pil_img = Image.open(test_img_file)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cv_img = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        face_img = pil_img.crop((x, y, x+w, y+h))
    else:
        face_img = pil_img
    gray_img = face_img.convert('L').resize(target_size)
    test_vec = np.asarray(gray_img, dtype=np.float32).flatten()
    centered_test = test_vec - mean_face.flatten()
    test_weights = np.zeros(eigenfaces.shape[1])
    for i in range(eigenfaces.shape[1]):
        test_weights[i] = np.dot(eigenfaces[:, i], centered_test)
    distances = np.zeros(training_weights.shape[1])
    for i in range(training_weights.shape[1]):
        distances[i] = euclidean_distance(test_weights, training_weights[:, i])
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    matched_name = names[min_idx]
    is_match = min_distance < threshold
    matched_img_path = os.path.join(folder_path, matched_name)
    matched_img = Image.open(matched_img_path)
    distance_df = pd.DataFrame({
        'Image': [name.split('.')[0] for name in names],
        'Distance': distances,
        'Match': ['✅ Best Match' if i == min_idx else 
                 ('✅ Within Threshold' if distances[i] < threshold else '❌ Too Far') 
                 for i in range(len(distances))]
    }).sort_values('Distance').reset_index(drop=True)
    exec_time = time.time() - start
    eigenface_data = {
        'eigenfaces': eigenfaces,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'mean_face': mean_face,
        'training_weights': training_weights,
        'test_weights': test_weights,
        'distance_analysis': distance_df
    }
    return matched_name, matched_img, exec_time, eigenface_data, is_match, min_distance
