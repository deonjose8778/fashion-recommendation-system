import streamlit as st
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import PIL
import pandas as pd

weights_path = '/Users/deonjose/Downloads/archive/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(weights=weights_path, include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

features_array, image_files = joblib.load('image_features.pkl')
styles_df = pd.read_pickle('styles_df.pkl')

if isinstance(features_array, list) or isinstance(features_array, tuple):
    features_array = np.array(features_array)

if features_array.size == 0:
    st.error("Features array is empty. Please check the feature extraction process.")
    st.stop()

if not np.issubdtype(features_array.dtype, np.number):
    st.error("Features array contains non-numeric data.")
    st.stop()

def extract_features(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def recommend_similar_images(uploaded_image, original_category, original_gender):
    img = PIL.Image.open(uploaded_image)
    features = extract_features(img)
    if features.size == 0:
        st.error("Failed to extract features from the uploaded image.")
        return []
    sim_scores = cosine_similarity([features], features_array)[0]
    top_indices = sim_scores.argsort()[-10:][::-1]  

    filtered_indices = [
        i for i in top_indices 
        if styles_df.iloc[i]['subCategory'] == original_category 
        and styles_df.iloc[i]['gender'] == original_gender
    ]
    top_filtered_indices = filtered_indices[:5]  

    return top_filtered_indices

st.title('Fashion Image Recommendation System')
st.write('Upload an image to get similar fashion items:')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file:
    try:
        uploaded_img = PIL.Image.open(uploaded_file)
        uploaded_img_id = int(uploaded_file.name.split('/')[-1].split('.')[0])
        original_category = styles_df[styles_df['id'] == uploaded_img_id]['subCategory'].values[0]
        original_gender = styles_df[styles_df['id'] == uploaded_img_id]['gender'].values[0]

        top_indices = recommend_similar_images(uploaded_file, original_category, original_gender)

        if len(top_indices) == 0:
            st.write("No similar images found.")
        else:
            recommendations = [image_files[i] for i in top_indices]

            image_ids = [int(img_path.split('/')[-1].split('.')[0]) for img_path in recommendations]

            recommendations_metadata = styles_df[styles_df['id'].isin(image_ids)]

            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            st.write('Recommended Similar Images:')
            num_cols = min(len(top_indices), 5)  
            cols = st.columns(num_cols)
            for i, (img_path, img_id) in enumerate(zip(recommendations, image_ids)):
                metadata = recommendations_metadata[recommendations_metadata['id'] == img_id].iloc[0]
                with cols[i]:
                    st.image(img_path, caption=img_path.split('/')[-1], use_column_width=True)
                    st.write(f"SubCategory: {metadata['subCategory']}")
    except IndexError:
        st.error("The uploaded image's ID does not exist in the metadata.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

for img_path in image_files:
    img_id = int(img_path.split('/')[-1].split('.')[0])
    if img_id not in styles_df['id'].values:
        st.error(f"Image {img_path} does not have corresponding metadata.")
