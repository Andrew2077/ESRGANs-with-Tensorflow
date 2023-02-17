import streamlit as st
from utils import functions as fn
from utils.functions import *
import tensorflow_hub as hub
import os


st.set_page_config(
    page_title="Super Resolution",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="auto",
)


if __name__ == "__main__":

    @st.cache_resource()
    def load_model():
        SAVE_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        return hub.load(SAVE_MODEL_PATH)

    model = load_model()

    st.title(":red[Super Resolution]")

    with st.sidebar.title("Settings"):
        ## Sidebar
        with st.container():
            ##Upload an image
            with st.container():
                st.title("**Settings**")
                uploaded_file = st.file_uploader(
                    ":red[Upload an Image]",
                    type=["jpg", "png"],
                    label_visibility="visible",
                    on_change=st.cache_data.clear(),
                )

                @st.cache_data()
                def uploade_image():
                    directory = os.getcwd()
                    image_dir = os.path.join(directory, uploaded_file.name)
                    # st.write(image_dir)
                    return fn.preprocess_image(image_dir)  # * the image is a tensor

                if uploaded_file is not None:
                    image = uploade_image()
                else:
                    image = None
                    # st.write("## **Please upload an image**")
                

            ##SLiders
            with st.container():
                width = st.slider(
                    "Select the width",
                    min_value=0,
                    max_value=1980,
                    value=0,
                    step=10,
                    label_visibility="visible",
                )
                height = st.slider(
                    "Select the height",
                    min_value=0,
                    max_value=1980,
                    value=0,
                    step=10,
                    label_visibility="visible",
                )

            ##Button
            #with st.container():
            with st.empty():
                space1 , space2 = st.columns(2)
                with space1:
                    enenchance_button = st.button("Enhance",)
                
                with space2:
                    st.checkbox('Auto Enhance', value=False)


    with st.container():
        uploaded, enhanced = st.columns(2)
        with uploaded:
            st.subheader("Uploaded image")
            if uploaded_file is not None:
                fig = fn.plot_image(image, title="")
                st.pyplot(fig)
                st.write(image.shape)
            else:
                st.write("**Please upload an image**")

        with enhanced:
            st.subheader("Enhanced image")
            if uploaded_file is not None:
                if uploaded_file is not None:
                    if enenchance_button:
                        #st.write(prepared_img.shape)
                        enhanced_img = fn.enhance_image(model, image)
                        fig = fn.plot_image(enhanced_img, title="")
                        st.image(enhanced_img)
                        st.pyplot(fig)
                        # st.write(enhanced_img.shape)
                        # st.sidebar.success("**The image is enchanted**")
                else:
                    st.sidebar.warning("**Enhance to see results**")
                    
        
