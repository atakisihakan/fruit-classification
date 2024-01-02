import streamlit as st
import base64, io, requests


st.set_page_config(page_title= 'Classify Fruits',
                   layout="centered", 
                   initial_sidebar_state= "auto",
                   page_icon=":watermelon:",)

def set_main_background(main_bg):
    '''
    Set a background image for the main page
    '''
    # set bg name
    main_bg_ext = "png"
    X = base64.b64encode(open(main_bg, "rb").read()).decode()

    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{X});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


def set_sidebar_background(side_bg):
    '''
    Set a background image for the sidebar
    '''
    side_bg_ext = 'png'
    X = base64.b64encode(open(side_bg, "rb").read()).decode()
    st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{X});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )


# Set sidebar background image
#set_sidebar_background('./Figures/fruits.jpg')

# Set main page background image
set_main_background('./Figures/main_background.jpg')

st.title(':rainbow[Classify Fruits]')

kaggle_link = '(https://www.kaggle.com/datasets/moltean/fruits/data)'
st.write(f"##### :gray[Classify select :red[20 fruits] \
            using a deep neural network model. \
            The model is built by fine-tuning *ResNet50* \
            pretrained on *ImageNet* data set. Specifically, \
            prediction layers of *ResNet50* were removed; \
            two custom dense layers were added on top which have been \
            further trained using the [Kaggle dataset]{kaggle_link}.]")

# Set foreground image
st.image('./Figures/fruits_bg.jpg',
         use_column_width= True)
# st.header("Header goes here...")
# st.subheader("Subheader goes here...")


col1, col2 = st.columns([0.5, 0.5])

with col1:
    # Upload image
    # clear existing url when an img is uploaded
    def clear_url():
        st.session_state['url_key'] = ''

    uploaded_image = st.file_uploader(
                            'Upload image',
                            type=['png','jpg'],
                            on_change= clear_url,
                            # call clear_url if img is uploaded
                            )
    
    st.markdown('')
    st.markdown("<h5 style='text-align: center; color: black;'>OR</h5>", 
                unsafe_allow_html=True)
    st.markdown('')


    # OR insert URL
    # You can use session state to store the value for the widget 
    # and clear it. Then, use the stored value in the rest 
    # of the script (not the “output” from the widget)
    def clear_widget():
        st.session_state['url_key'] = st.session_state['widget_key']
        st.session_state['widget_key'] = ''

    st.text_input('Enter URL of image',
                    key= 'widget_key',
                    on_change= clear_widget, 
                    # calling clear_widget does:
                    # 1) read st.session_state['widget_key'] which is the input text
                    # 2) store it in st.session_state['url_key']
                    # 3) clear widget field by st.session_state['widget_key'] = ''
                    )
    url_image = st.session_state.get('url_key', '')
    # .get('url_key', '') ensures 
    # if 'url_key' not in st.session_state dic 
    # it will be created and set to empty string


with col2:
    st.markdown('')
    st.markdown('')
    img_holder = st.empty()
    prediction_holder = st.empty()


# Footer stuff
st.markdown("***")
st.markdown(
    '`Created by` [hakanatakisi](https://www.google.com/) | \
        `Code` [GitHub](https://www.google.com/)')


# Loading modules take considerable time ~10 sec
# So, load them after main elements of the webpage finished 
# Load modules while user wandering around the webpage
import keras
from prediction import predict, remove_background, load_model

# load_model() fn is cached, so call it now 
# (while the user is wandering around the webpage) 
# to save time later when making first prediction
# if you don't call it now, 
# the first prediction will take ~10 sec longer
# but the subsequent predictions will be normal
load_model()



if url_image != '':
    uploaded_image = io.BytesIO(requests.get(url_image).content)

# Prediction stuff
if uploaded_image is not None:

    # Hide file name and size info on UI
    st.markdown("<style>.uploadedFile {display: none}<style>",
                unsafe_allow_html=True)

    # Display original image
    # resized_image = keras.utils.load_img(uploaded_image,
    #                         target_size= (200, 200),
    #                         keep_aspect_ratio= True, # crop in the center
    #                         )
    
    # img_holder.image(resized_image, 
    #                  #use_column_width=True,
    #                  )
    img_holder.image(uploaded_image,
                     use_column_width= True)

    # This is the placeholder of 
    # where prediction will be display i.e.
    # right under the image after calc. is done
    #prediction_holder = st.empty()

    with col2:
        # Temporary spinner while calculating
        with st.spinner("Classifying..."):
            bg_removed = remove_background(uploaded_image)
            pred_class_label, confidence = predict(bg_removed)

    # Print prediction and confidence
    # txt = f'##### :orange[***{pred_class_label}***]' +\
    #   f':gray[ (confidence: {confidence}%)]'
    # prediction_holder.success(txt, icon= '🤖')

    prediction_holder.write(f'#### :orange[***{pred_class_label}***]  \
                                :gray[({confidence}% :robot_face:)]')