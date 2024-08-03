Fashion Recommendation System

This project aims to recommend fashion products based on image similarity and metadata using machine learning techniques.

Objective

Develop a system to recommend similar fashion items based on image features and metadata.

Data

styles.csv: Metadata including id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, and productDisplayName.
image_features.pkl: Precomputed image features using VGG16.
features_by_gender_subcategory.pkl: Precomputed features by gender and subCategory.
styles_df.pkl: Combined dataset.
vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5: Pre-trained VGG16 weights.
Model

Feature Extraction: VGG16 model extracts image features.
Similarity Measurement: Cosine similarity compares features to find similar items.
Filtering: Recommendations are filtered by subCategory and gender.
Results

Accurate image-based recommendations.
Filtering ensures relevant suggestions based on metadata.
Data Preprocessing

Handled missing values and encoded categorical variables to improve accuracy.

Deployment

The project is deployed as a web app using Streamlit, allowing users to upload images and receive instant recommendations.

How to Run the App

Clone the repository.
Install dependencies: pip install -r requirements.txt.
Run the Streamlit app: streamlit run app.py.

Usage

Upload an image to the web app. The model will recommend similar fashion items based on the input image and metadata.

Acknowledgments

This project uses pandas, numpy, scikit-learn, Keras, and Streamlit. Special thanks to the dataset providers and the open-source community for their contributions.