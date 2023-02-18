import streamlit as st
from utils import functions as fn
from utils.functions import *
import tensorflow_hub as hub
import os


st.set_option("deprecation.showPyplotGlobalUse", False)

def _local_deBuffer():
     global new_image_uploaded 
     new_image_uploaded = False


def callbacks():
     global new_image_uploaded 
     st.cache_data.clear()
     if (os.path.exists(os.path.join(os.getcwd(), "enhanced_image.jpg"))):
        os.remove(os.path.join(os.getcwd(), "enhanced_image.jpg"))
     new_image_uploaded = True

@st.cache_resource()
def tf_better_memory():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


st.set_page_config(
    page_title="Super Resolution",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """<style>
        .element-container:nth-of-type(1) button {
            height: 2em;
            margin: 0em 0em 0em 0em;
            padding: 0.1em 1em 0.1em 1em;
            font-size: 2em;
            font-weight: bold;
        }
        .element-container:nth-of-type(1) div {
            colro: red;
        }
        </style>""",
    unsafe_allow_html=True,
)
tf_better_memory()


if __name__ == "__main__":

    @st.cache_resource()
    def load_model_TF():
        SAVE_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        return hub.load(SAVE_MODEL_PATH)
    
 

    model_loader = st.checkbox("load model", value=False)
    if model_loader:
        model = load_model_TF()

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
                    on_change=callbacks()
                )
                selected_model = st.selectbox("Select a model", ["ESRGAN_x4_TF", 'RealESRGAN_x2', 'RealESRGAN_x4', 'RealESRGAN_x8'])
                # if model_loader == 'ESRGAN_x4_TF':
                st.sidebar.success(f"**{selected_model} is loaded**")
                

            ##SLiders
                @st.cache_data()
                def uploade_image():
                    directory = os.getcwd()
                    image_dir = os.path.join(directory, uploaded_file.name)
                    with open(image_dir, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    new_image_uploaded = True
                    return fn.preprocess_image(image_dir), new_image_uploaded  # * the image is a tensor

                if uploaded_file is not None:
                    image, new_image_uploaded = uploade_image()
                    #enhanced_img = enhance_img_streamlit()
                else:
                    image = None
                    enhanced_img = None
                    #new_image_uploaded = False
                    # st.write("## **Please upload an image**")
                    
            ##Button
            
            with st.empty() as k:
                space1, space2 = st.columns(2)
                with space1:
                    enhance_button = st.button(
                        "Enhance",
                        key="enhance",
                        #on_click=_local_deBuffer()
                    )
                with space2:
                    if uploaded_file is None:
                        st.button("Download", disabled=True)

            auto_enhance = st.checkbox("Auto Enhance", value=False)
            if auto_enhance:
                st.markdown(
                    """<style>
                        .element-container:nth-of-type(5) div {
                            color: green;;  })""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """<style>
                        .element-container:nth-of-type(5) div {
                            color: red;;  })""",
                    unsafe_allow_html=True,
                )

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
                @st.cache_data()
                def enhance_img_streamlit():
                    return fn.enhance_image(model,image)
                
                if enhance_button or auto_enhance:
                    # st.write(prepared_img.shape)
                    with st.spinner("Enhancing the image..."):
                        enhanced_img = enhance_img_streamlit()
                        save_image(enhanced_img, filename="enhanced_image")
            
            
            if (os.path.exists(os.path.join(os.getcwd(), "enhanced_image.jpg"))) and (new_image_uploaded is True):
                enhanced_img = Image.open("enhanced_image.jpg")
                fig = fn.plot_image(enhanced_img, title="")
                st.pyplot(fig)
                if enhanced_img.mode == "RGB":
                    st.write(tuple((enhanced_img.size[0], enhanced_img.size[1], 3)))
                st.sidebar.success("**The image is enhanced**")
                with space2:
                    save_image(enhanced_img, filename="enhanced_image")
                    with open("enhanced_image.jpg", "rb") as f:
                        btn = st.download_button(
                            label="Download",
                            data=f,
                            file_name="enhanced_image.jpg",
                            mime="image/jpg",
                        )

            else:
                st.sidebar.warning("**Enhance to see results**")
