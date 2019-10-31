# Face and Emotion Recognition
This software recognizes person's faces and their corresponding emotions from a video or webcam feed. Powered by OpenCV, Dlib, face_recognition and Deep Learning.

## Dependencies
- Opencv
- Dlib
- Keras

## Usage
- `test` folder contain images or video that we will feed to the model.
- `images` folder contain only images of person face to perform face recognition.
- `models` contain the pre-trained model for emotion classifier.
- `emotion.py` can to run to classify emotions of person's face.
- `face-rec-emotion.py` can recognise faces and classify emotion at a time.
- face_recognition library uses the FaceNet Implementation for face recognition.

`python emotion.py`

`python face-rec-emotion.py`


## To train new models for emotion classification

- Place the train dataset into `images` folder
- Run the train_emotion_classification.py file:
`python train_emotion_classifier.py`