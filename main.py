import os
import tensorflow
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

from display_images import displaying_images_in_pagination
# from upload_image import get_displaying_images


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap');
    .st-emotion-cache-1v0mbdj.e115fcil1 img {
        width: 100%;

            height: 200px;

    }

    .st-emotion-cache-1kyxreq.e115fcil2 {
        display: block;
    }
    .element-container.st-emotion-cache-1lis4bp.e1f1d6gn4 .st-emotion-cache-9aoz2h.e1vs0wn30 .st-emotion-cache-1kyxreq.e115fcil2 .st-emotion-cache-1v0mbdj.e115fcil1 img {
        display: flex;
        width: 250px;
    }
    body {
        font-family: 'Poppins', sans-serif!important;
    }
    .st-emotion-cache-1r4qj8v{
        background: #f3f7fd;  /* fallback for old browsers */

    }
    h1 , h2 ,h3 ,h4 ,h5 ,h6{
        font-family: 'Poppins', sans-serif!important;
    }
    p{
        font-family: 'Poppins', sans-serif!important;  
    }
    .st-emotion-cache-1629p8f.e1nzilvr2 h2 {
        font-size: 24px;
        padding-top: 0;
        
    }
    .st-emotion-cache-1629p8f.e1nzilvr2 h1 {
        padding-top: 0;
        font-size: 20px;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 80rem;
    }
    
    h1#monthly-timeline .st-emotion-cache-zt5igj {
    padding: 0 20px;
    min-height: 60px;
    border-bottom: 1px solid #f2edff;
    }

    h1#daily-timeline .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }
    h1#most-busy-day .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }
    h1#most-busy-month .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }
    h1#wordcloud .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }
        h1#most-busy-users .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }

    h1#weekly-activity-map .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }
    h1#emoji-chart .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }
    h1#emoji-analysis .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }
    h1#most-online-person .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }


    h1#most-online-person .st-emotion-cache-zt5igj {
        padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
    }

    h1#most-common-words .st-emotion-cache-zt5igj {
        
    padding: 0 20px;
        min-height: 60px;
        border-bottom: 1px solid #f2edff;
        background: #ffffff;
        -webkit-border-radius: 5px;
        -moz-border-radius: 5px;
        border-radius: 5px 5px 0 0;
        -webkit-box-shadow: 0px 0px 12px 0px rgba(55, 73, 72, 0.3);
        -moz-box-shadow: 0px 0px 12px 0px rgba(55, 73, 72, 0.3);
        box-shadow: 0px 0px 12px 0px rgba(55, 73, 72, 0.3);
    }

    .st-emotion-cache-zt5igj {
        position: relative;
        left: 0;
        width: 100%;
    }
    .st-emotion-cache-10trblm {
        margin-left: calc(0rem);
    }
    h1#most-common-words {
        padding-bottom: 0;
    }
    .st-emotion-cache-ocqkz7.e1f1d6gn5 .st-emotion-cache-1cwt8z2.e1f1d6gn3 {
        width: calc(25% - 1rem);
        flex: 1 1 calc(25% - 1rem);
    }

    .st-emotion-cache-ocqkz7.e1f1d6gn5 .st-emotion-cache-1cwt8z2.e1f1d6gn3 .st-emotion-cache-1kyxreq.e115fcil2 {
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


feature_list = np.array(pickle.load(open('embedded_images.pkl','rb')))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices, distances


if __name__ == '__main__':
    # file upload -> save
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            # display the file
            display_image = Image.open(uploaded_file)
            st.image(display_image)
            # feature extract
            features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
            #st.text(features)
            indices, distances = recommend(features,feature_list)
            print('distances   ddddddd', distances)
            displaying_images = []
            for num in range(15):
                distance_in_int = int(distances[0][num] * 100)
                if distance_in_int < 80:
                    displaying_images.append(indices[0][num])

        print('displaying images', displaying_images)

        displaying_images_in_pagination(displaying_images)
    else:
        st.header("Some error occured in file upload")
