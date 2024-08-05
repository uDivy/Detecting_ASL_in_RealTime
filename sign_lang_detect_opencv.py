#!/usr/bin/env python
# coding: utf-8

# ## Import and Install Dependencies

# In[1]:


# !pip install tensorflow==2.4.1


# In[2]:


#!pip install tensorflow-gpu==2.4.1


# In[3]:


# !pip install mediapipe


# In[4]:


# !pip install opencv-python


# In[5]:


# !pip install scikit-learn


# In[6]:


# !pip install matplotlib


# In[7]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# ## Keypoints using MP Holistic

# In[8]:


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing Utilities


# In[9]:


def mediapipe_detecttion(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color Conversion 
    image.flags.writeable = False # Image is no longer writable
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# In[10]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return 


# In[11]:


# mp_drawing.draw_landmarks?


# In[12]:


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


# In[13]:


# cap = cv2.VideoCapture(0)
# # Set mediapipe model

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()

#         # Make Detections
#         image, results = mediapipe_detecttion(frame, holistic)
        
#         # Draws Landmarks
#         draw_styled_landmarks(image, results)

#         # to display the frame
#         cv2.imshow('OpenCV Feed', image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# ## Extract Keypoint Values

# In[14]:


def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z]for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)

    pose = np.array([[res.x, res.y, res.z, res.visibility]for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    lh = np.array([[res.x, res.y, res.z]for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)

    rh = np.array([[res.x, res.y, res.z]for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, face, lh, rh])


# ## Setup Folders for Collection

# In[15]:


# len(extract_keypoints(results))


# In[57]:


# DATA_PATH = os.path.join('MP_Data') # path for exported datas, numpy arrays
actions = np.array(['hello', 'thanks', 'iloveyou']) # actions that we are gonna detect
# no_sequence = 30 # 30 videos worth of data 
# sequence_length = 30 # videos are going to be 30 frames in length


# In[17]:


# for action in actions:
#     for sequence in range(no_sequence):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass
    


# ## Collect Keypoints Values for Training and Testing

# In[18]:


# cap = cv2.VideoCapture(0)
# # Set mediapipe model

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     # Loop through actions
#     for action in actions:
        
#         # loop through sequences aka videos
#         for sequence in range(no_sequence):
#             # loop through video length aka sequence length
#             for frame_num in range(sequence_length):
                
#                 ret, frame = cap.read()
                
#                 # Make Detections
#                 image, results = mediapipe_detecttion(frame, holistic)

#                 # Draws Landmarks
#                 draw_styled_landmarks(image, results)
                
#                 # Apply wait logic
#                 if frame_num == 0:
#                     cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                                (0, 255, 0), 1, cv2.LINE_AA)
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                (0, 0, 255), 1, cv2.LINE_AA)
#                     # to display the frame
#                     cv2.imshow('OpenCV Feed', image)
#                     cv2.waitKey(2000)
#                 else:
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                (0, 0, 255), 1, cv2.LINE_AA)
#                     # to display the frame
#                     cv2.imshow('OpenCV Feed', image)
                    
#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)
                    
                
                
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break

#     cap.release()
#     cv2.destroyAllWindows()


# In[19]:


# cap.release()
# cv2.destroyAllWindows()


# ## Preprocess Data and Create Labels and Features

# In[20]:


# from sklearn.model_selection import train_test_split


# In[21]:


# !pip install typing
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'


# In[22]:


# import tensorflow as tf
# print(tf.__version__)

# import h5py
# print(h5py.__version__)


# In[23]:


# from tensorflow.keras.utils import to_categorical # to convert into one-hot encoded data


# In[24]:


# label_map = {label:num for num, label in enumerate(actions)}
# label_map


# In[25]:


# sequences, labels = [], []
# for action in actions:
#     for sequence in range(no_sequence):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])


# In[26]:


# np.array(sequences).shape # 90 videos each with 30 frame each and each frame has 1662 data points


# In[27]:


# np.array(labels).shape # for first 30 video label is 0, and 1 and 2 respectively for next videos i.e target values


# In[28]:


# X = np.array(sequences)
# X.shape


# In[29]:


# y = np.array(labels)
# y


# In[30]:


# y = to_categorical(labels).astype(int)
# y


# In[31]:


# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[32]:


# x_train.shape


# In[33]:


# x_test.shape


# ## Build and Train LSTM Neural Network

# In[34]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense # LSTM for temporal component for Action Detection
from tensorflow.keras.callbacks import TensorBoard # login for monitoring


# In[35]:


# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)


# In[36]:


model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))


# In[37]:


# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[38]:


# model.fit(x_train, y_train, epochs=1500, callbacks=[tb_callback])


# In[39]:


# model.summary()


# ## Make Predictions

# In[40]:


# res = model.predict(x_test)
# res.shape


# In[41]:


# actions[np.argmax(res[3])]


# In[42]:


# actions[np.argmax(y_test[3])]


# ## Save Weights

# In[43]:


# model.save('action.h5')
from tensorflow.keras.models import load_model

# Load the model from a saved model file
model = load_model('action.h5')


# ## Evaluation using Confusion Matrix and Accuracy

# In[44]:


# from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[45]:


# yhat = model.predict(x_test)
# yhat.shape


# In[46]:


# y_test


# In[47]:


# ytrue = np.argmax(y_test, axis=1).tolist()
# ytrue


# In[48]:


# yhat = np.argmax(yhat, axis=1).tolist()
# yhat


# In[49]:


# import seaborn as sns


# In[50]:


# cm = multilabel_confusion_matrix(ytrue, yhat)
# cm #[Tn]


# In[51]:


# accuracy_score(ytrue, yhat)


# ## Test in Real Time

# In[52]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[65]:


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# In[73]:


# New detection variables
sequence = [] # concatenate 30 frames of data to generate prediction
sentence = []
threshold = 0.9
predictions = []

cap = cv2.VideoCapture(0)
# Set mediapipe model

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Make Detections
        image, results = mediapipe_detecttion(frame, holistic)
        
        # Draws Landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
        # Vis logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]: # if current action is not same as last
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                    
            if len(sentence) > 5:
                sentence = sentence[-5:]
        
            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
#         cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
#         cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # to display the frame
        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[70]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:




