import cv2
import os
from tqdm import tqdm

ALL_EMOTIONS = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]

# RAW_TRAIN_DIR = 'F:/Python/FER/MIXED/raw_train' # Directory for raw train images
RAW_TEST_DIR = 'raw_test' # Directory for raw test images
TRAIN_DIR = 'train' # Directory for processed train images
TEST_DIR = 'test' # Directory for processed test images

# --------------------------------------------------------------------------------------------------------
# Given a rgb image crop face portion and then performs histogram equalization
# Returns Grayscale face image
# Should contain image of exactly one person
# Need to apply cropping of image in recognizer.py according to choosen IMG_SIZE
def preprocess_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceDet_one = cv2.CascadeClassifier("face_classifiers/haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("face_classifiers/haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("face_classifiers/haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("face_classifiers/haarcascade_frontalface_alt_tree.xml")

    face_one = faceDet_one.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face_one) == 1:
        facefeatures = face_one
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        output_img = cv2.equalizeHist(gray_img)
        return output_img

    for (x, y, w, h) in facefeatures:
        gray_img = gray_img[y:y+h, x:x+w]
        output_img = cv2.equalizeHist(gray_img)
        return output_img

# --------------------------------------------------------------------------------------------------------
# Given the labeled image in RAW_TRAIN_DIR it creates seven emotion folders in TRAIN_DIR
# For jaffe databse
def create_train_images():
    # Creating all emotions folder in the training directory
    for emotion in ALL_EMOTIONS:
        emotion_dir = os.path.join(TRAIN_DIR, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

    # Processing image and saving in train directory
    for part in os.listdir(RAW_TRAIN_DIR+'/emotion'):
        for sessions in os.listdir(RAW_TRAIN_DIR+'/emotion/'+part):
            for files in os.listdir(RAW_TRAIN_DIR+'/emotion/'+part+'/'+sessions):

                file_path = RAW_TRAIN_DIR+'/emotion/'+part+'/'+sessions+'/'+files
                label = open(file_path, 'r')
                idx = int(float(label.readline()))
                if(idx == 2):
                    continue    # Ignoring contempt emotion
                if(idx > 2): 
                    idx -= 1    # Since contempt is ignored idx should be 1 minused

                emotion = ALL_EMOTIONS[idx]
                img_path = RAW_TRAIN_DIR+'/images/'+part+'/'+sessions
                neutral_path = os.listdir(img_path)[0]
                emotion_path = os.listdir(img_path)[-1]
                neutral_img_path = img_path+'/'+neutral_path
                emotion_img_path = img_path+'/'+emotion_path
                
                if(neutral_img_path[-3:] == 'png'):
                    print('1', neutral_img_path, neutral_img_path[-3:])
                    neutral_img = cv2.imread(neutral_img_path)
                    neutral_img = preprocess_img(neutral_img)
                    cv2.imwrite(TRAIN_DIR+'/neutral/'+neutral_path, neutral_img)

                if(emotion_img_path[-3:] == 'png'):
                    print('2', emotion_img_path, emotion_img_path[-3:])
                    emotion_img = cv2.imread(emotion_img_path)
                    emotion_img = preprocess_img(emotion_img)
                    cv2.imwrite(TRAIN_DIR+'/'+emotion+'/'+emotion_path, emotion_img)

# --------------------------------------------------------------------------------------------------------
# Given the unlabeled image in RAW_TEST_DIR it saves processed images in TEST_DIR
# Need to apply cropping of image in recognizer.py according to choosen IMG_SIZE
def create_test_images():
    for files in os.listdir(TEST_DIR):
        os.remove(TEST_DIR+'/'+files)
    for img in tqdm(os.listdir(RAW_TEST_DIR)):
        img_path = os.path.join(RAW_TEST_DIR, img)
        img_data = cv2.imread(img_path)
        img_data = preprocess_img(img_data)
        save_img_path = os.path.join(TEST_DIR, img)
        cv2.imwrite(save_img_path, img_data)

# --------------------------------------------------------------------------------------------------------
# create_train_images() # Training data is copied from other prepocessors so dont apply here
create_test_images()