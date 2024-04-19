import uuid
from pathlib import Path

import numpy as np
import av
import cv2
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import tensorflow as tf
import keras
from pydub import AudioSegment, effects
import librosa
import noisereduce as nr

index = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def get_ice_servers():
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # perform edge detection
    img = cv2.cvtColor(cv2.Canny(img, 100, 200))

    return av.VideoFrame.from_ndarray(img, format="bgr24")


RECORD_DIR = Path("./records")
RECORD_DIR.mkdir(exist_ok=True)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0
    max_frames = 80

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        frame = cv2.resize(frame, (100, 56))
        
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)

        frame_count += 1

    cap.release()

    return frames

def process_audio(video_path, total_length=300000, frame_length=2048, hop_length=512):
    rms = []
    zcr = []
    mfcc = []

    _, sr = librosa.load(path = video_path, sr = None)
    rawsound = AudioSegment.from_file(video_path)
    normalizedsound = effects.normalize(rawsound, headroom = 0)
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')
    xt, index = librosa.effects.trim(normal_x, top_db=30)
    xt_length = len(xt)
    if xt_length < total_length:
        pad_length = total_length - xt_length
        padded_x = np.pad(xt, (0, pad_length), 'constant')
    elif xt_length > total_length:
        padded_x = xt[:total_length]
    else:
        padded_x = xt
    final_x = nr.reduce_noise(padded_x, sr=sr)

    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length)
    f2 = librosa.feature.zero_crossing_rate(final_x , frame_length=frame_length, hop_length=hop_length, center=True)
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=13, hop_length = hop_length)

    rms.append(f1)
    zcr.append(f2)
    mfcc.append(f3)

    f_rms = np.asarray(rms).astype('float32')
    f_rms = np.swapaxes(f_rms,1,2)
    f_zcr = np.asarray(zcr).astype('float32')
    f_zcr = np.swapaxes(f_zcr,1,2)
    f_mfccs = np.asarray(mfcc).astype('float32')
    f_mfccs = np.swapaxes(f_mfccs,1,2)

    X = np.concatenate((f_zcr, f_rms,f_mfccs), axis=2)
    print(X.shape)
    return X


def recognize(in_file):

    frames = process_video(str(in_file))
    audio = process_audio(str(in_file))
    with open('model.json', 'r') as json_file:
        json_model = json_file.read()
    model = keras.models.model_from_json(json_model)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['categorical_accuracy'])
    model.load_weights('model_weights.h5')

    input_data = np.array([frames])
    input_data = input_data.reshape(input_data.shape[0], 80, 5600)
    # Передача підготовлених даних моделі
    predictions = model.predict(input_data)
    max_index = np.argmax(predictions)
    
    # Вибір відповідної емоції зі списку `index`
    emotion = index[max_index]
    
    st.write(f"Video detected emotion: {emotion}")

    with open('audio.json', 'r') as json_file:
        json_model = json_file.read()
    model = keras.models.model_from_json(json_model)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['categorical_accuracy'])

    model.load_weights('audio_weights.h5')

    predictions = model.predict(audio)
    max_index = np.argmax(predictions)
    
    # Вибір відповідної емоції зі списку `index`
    emotion = index[max_index]
    
    st.write(f"Audio detected emotion: {emotion}")

def app():
    if "prefix" not in st.session_state:
        st.session_state["prefix"] = str(uuid.uuid4())
    prefix = st.session_state["prefix"]
    in_file = RECORD_DIR / f"{prefix}_input.mp4"
    out_file = RECORD_DIR / f"{prefix}_output.mp4"

    def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(
            str(in_file), format="mp4"
        )

    def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder(str(out_file), format="mp4")

    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={
            "video": True,
            "audio": True,
        },
        video_frame_callback=video_frame_callback,
        in_recorder_factory=in_recorder_factory,
        out_recorder_factory=out_recorder_factory,
    )

    st.button(key='rec',label='Recognize', on_click=lambda:recognize(in_file))

    if in_file.exists():
        #frames = process_video(str(out_file))
        st.download_button(
            "Download the recorded video without video filter",
            in_file.read_bytes(),
            f"{prefix}_input.mp4",
            mime="video/flv",
        )
    if out_file.exists():
        st.download_button(
            "Download the recorded video with video filter",
            out_file.read_bytes(),
            f"{prefix}_output.mp4",
            mime="video/flv",
        )


if __name__ == "__main__":
    app()