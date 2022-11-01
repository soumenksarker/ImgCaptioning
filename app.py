from random import sample
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
import streamlit as st
import pickle
from PIL import Image
#pickle.load(open('energy_model.pkl', 'rb'))
#vocab = np.load('w2i.p', allow_pickle=True)

new_dict = pickle.load(open('w2i.p', 'rb'))
inv_dict= pickle.load(open('i2w.p', 'rb'))
#vocab = vocab.item()
#inv_vocab = {v:k for k,v in vocab.items()}
print("+"*50)
print("vocabulary loaded")
embedding_size = 128
vocab_size = len(new_dict)
max_len = 36
MAX_LEN = max_len
image_model = Sequential()
image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))
language_model = Sequential()
language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))
conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model=load_model('model.h5')
model.load_weights('mine_model_weights.h5')

print("="*150)
print("MODEL LOADED")
st.title("img_captioning_app")
#st.text("Build with Streamlit and OpenCV")
if "photo" not in st.session_state:
	st.session_state["photo"]="not done"

c2, c3 = st.columns([2,1])
def change_photo_state():
	st.session_state["photo"]="done"
#incept_model = ResNet50(include_top=True)
#last = incept_model.layers[-2].output
#resnet = Model(inputs = incept_model.input,outputs = last)
resnet = load_model('resnet.h5')
print("="*150)
print("RESNET MODEL LOADED")

@st.cache
def load_image(img):
	im = Image.open(img)
	return im
activities = ["generate_caption","About"]
choice = st.sidebar.selectbox("Select Activty",activities)
uploaded_photo = c2.file_uploader("Upload Image",type=['jpg','png','jpeg'], on_change=change_photo_state)
camera_photo = c2.camera_input("Take a photo", on_change=change_photo_state)
if choice == 'generate_caption':
    st.subheader("Face Detection") 
    if st.session_state["photo"]=="done":
        if uploaded_photo:
            our_image= load_image(uploaded_photo)
        elif camera_photo:
            our_image= load_image(camera_photo)
        elif uploaded_photo==None and camera_photo==None:
            our_image= load_image('image.jpg')
        our_image = np.array(our_image.convert('RGB'))
        ima= cv2.cvtColor(our_image, cv2.COLOR_BGR2RGB)
        ima = cv2.resize(ima, (224,224))
        ima = np.reshape(ima, (1,224,224,3))
        test_feature = resnet.predict(ima).reshape(1,2048)
        print("="*50)
        print("Predict Features")
        text_inp = ['startofseq']
        count = 0
        caption = ''
        while count < 25:
            count += 1
            encoded = []
            for i in text_inp:
                encoded.append(new_dict[i])
            encoded = [encoded]
            encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)
            prediction = np.argmax(model.predict([test_feature, encoded]))
            sampled_word = inv_dict[prediction]
            caption = caption + ' ' + sampled_word
            if sampled_word == 'endofseq':
                break
            text_inp.append(sampled_word)
        st.success(caption[:-8])
