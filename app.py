import streamlit as st
import base64, io, requests


# Input figures for design
background_img_path = './Figures/main_background.jpg'
foreground_img_path = './Figures/fruits_foreground.jpg'

# Page settings
st.set_page_config(page_title= 'Classify Fruits',
                   layout= 'centered', 
                   initial_sidebar_state= 'expanded',
                   page_icon= ':watermelon:',)

# Set main page background image
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
set_main_background(background_img_path)

# Set sidebar background image
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
#set_sidebar_background('./Figures/fruits.jpg')


st.title(':rainbow[Classify Fruits]')

link = '(https://www.kaggle.com/datasets/moltean/fruits/data)'
st.write(f"##### :gray[Classify :red[20 fruits] shown below \
            using a deep neural network model. \
            The model is built by fine-tuning *ResNet50* \
            pretrained on *ImageNet* data set. Specifically, \
            prediction layers of *ResNet50* were removed; \
            two custom dense layers were added on top \
            which have been further trained using \
            the [Kaggle dataset]{link}.]")

# Set foreground image
st.image(foreground_img_path,
         use_column_width= True)

st.write(':gray[**Note**: The model has been trained \
         on relatively a small dataset coming from \
         a single source. Thus, it works best \
         when the input image contains \
         a single fruit (or as few as possible),  \
         unpeeled and has a plain background.]')


# Upload image or insert URL on sidebar
with st.sidebar:

    # clear existing url, if any, when an img is uploaded
    def clear_url():
        st.session_state['url_key'] = ''

    uploaded_image = st.file_uploader(
                            'Upload image',
                            type=['png','jpg'],
                            on_change= clear_url,
                            # call clear_url if img is uploaded
                            )

    # Write centered "OR" between upload and URL
    st.write("<h5 style='text-align: center; \
             color: gray;'>OR</h5>",
             unsafe_allow_html= True)

    # Use session state to store the value for the widget 
    # and clear it. Then, use the stored value in the rest 
    # of the script (not the “output” from the widget)
    def clear_widget():
        st.session_state['url_key'] = st.session_state['widget_key']
        st.session_state['widget_key'] = ''

    st.text_input(
        'Enter image URL',
        key= 'widget_key',
        on_change= clear_widget, 
        # calling clear_widget does:
        # 1) read st.session_state['widget_key'] i.e. input text
        # 2) store it in st.session_state['url_key']
        # 3) clear widget by st.session_state['widget_key'] = ''
        )
    url_image = st.session_state.get('url_key', '')
    # .get('url_key', '') ensures 
    # if 'url_key' not in st.session_state dic 
    # it will be created and set to empty string

    
# Footer stuff
st.write('***')
st.write(
    '`Created by` [hakanatakisi](https://github.com/atakisihakan) | \
        `Code` [GitHub](https://github.com/atakisihakan/fruit-classification)')


# Loading modules take considerable time ~10 sec
# So, load them after main elements of the webpage finished 
# Load modules while user wandering around the webpage
from prediction import predict, remove_background, load_model

# load_model() fn is cached, so call it now 
# (while the user is wandering around the webpage) 
# to save time later when making first prediction
# if you don't call it now, 
# the first prediction will take ~10 sec longer
# but the subsequent predictions will be normal
load_model()


# Load image into memory if URL not empty
if url_image != '':
    img_content = requests.get(url_image).content
    uploaded_image = io.BytesIO(img_content)


if uploaded_image is not None:

    # Hide file name and size info on UI
    st.write("<style>.uploadedFile {display: none}<style>",
             unsafe_allow_html=True)

    # Set elements in the sidebar
    with st.sidebar:

        # This is the placeholder of 
        # where prediction will be display i.e.
        # on top of the image after calc. is done
        prediction_holder = st.empty()

        # Placeholder for temp spinner while calculating
        spinner_holder = st.empty()

        # Display original image
        st.image(uploaded_image, use_column_width= True)

        # Temp spinner while calculating
        with spinner_holder, st.spinner("Classifying..."):
            bg_removed = remove_background(uploaded_image)
            pred_class_label, confidence = predict(bg_removed)
        

        # Print prediction and confidence
        prediction_holder.write(
            f'## :green[***{pred_class_label}***] \
            :gray[({confidence}% :robot_face:)]')