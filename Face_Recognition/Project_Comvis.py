import cv2 as cv
import os
import numpy as np
import math

# Membuat pre-defined classifier yang dibasiskan pada haar-cascade. Pada
# xml file tersebut berisi poin poin penting yang membuat sebuah mesin dapat mengenali bentuk
# rupa sesuatu, dalam hal ini wajah seseorang. Haar-Cascade pun tidak hanya bisa mengenali
# wajah seseorang, dia pun bisa mengenali bentuk-bentuk lain.
face_cascade = cv.CascadeClassifier (cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ini menetapkan folder yang akan kita train dan test
def get_path_list(root_path):

    img_path = os.listdir(root_path)

    return img_path

# Setelah membuat classifier object, kita melakukan iterasi terhadap semua foto
# yang akan di train. Disini kita menetapkan folder yangakan dipakai untuk di training
def get_class_names(root_path, train_names):

    image_dir = []
    class_list = []

    for index, name_train in enumerate(train_names):
        full_name_path = root_path + name_train

        for image_path in os.listdir(full_name_path):
            full_image_path = full_name_path + '/' + image_path
            image_dir.append(full_image_path)
            class_list.append(index)

    return image_dir, class_list

# Pendeteksian wajahini bisa dicapai dengan method detectMultiScale(). method ini berguna untuk
# mendeteksi lebih dari 1 wajah pada 1 foto.  detectMultiScale sendiri memiliki beberapa parameter penting yang
# perlu dikonfigurasi sesuai dengan kebutuhan. Yang pertama adalah scaleFactor yang
# digunakan untuk memperkecil source image yang ada pada setiap skala tertentu. minNeighbors
# digunakan sebagai threshold agar sebuah dimensi X dan Y ditetapkan sebagai sebuah wajah
def detect_faces_and_filter(image_list, image_classes_list):
    face_filter = []
    face_class_list = []

    for index, (image, image_class) in enumerate (zip(image_list, image_classes_list)):
        img_gray = cv.imread(image, 0) #Ubah ke Grayscale
        detected_faces = face_cascade.detectMultiScale(img_gray, 1.2, 5)

        if(len(detected_faces) < 1):
            continue

        for face_rect in detected_faces:
            x, y, h, w = face_rect
            image_face = img_gray[y:y+h, x:x+w] #Cropping
            face_filter.append(image_face)
            face_class_list.append(image_class)
    

    return face_filter, face_class_list

def detect_test_faces_and_filter(image_list):
    face_image_filtered = []
    face_rect = []

    for index, image in enumerate(image_list):
        img_gray = cv.imread(image, 0)
        detected_faces = face_cascade.detectMultiScale(img_gray, 1.2, 5)

        if(len(detected_faces)< 1):
            continue

        for faces_rect in detected_faces:
            x,y,h,w = faces_rect
            image_face = img_gray[y:y+h, x:x+w]
            face_image_filtered.append(image_face)
            face_rect.append((x,y,h,w))

    return face_image_filtered, face_rect


# Disini kita membuat recognizer object. Setelah mendeteksi adanya wajah, saatnya membuat
# sebuah object untuk mengenali wajah tersebut milik siapa. Disini saya memakai bantuan
# LBPH (Local Binary Pattern Histogram) algorithm. Disni saya men-training recognizer
# tersebut berdasarkan data training yang sudah kita dapatkan tadi
def train(train_face_grays, image_classes_list):

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))

    return face_recognizer


# Bagian ini saatnya kita mencoba hasil recognizer yang sudah dilatih. Disini kita bisa mengecek
# seberapa yakin recognizer tersebut dalam memprediksi gambar yang diberikan. Semakin kecil 
# error marginnya maka semakin yakin recognizer tersebut bahwa hasil yang diprediksi itu benar.
def get_test_images_data(test_root_path):
    
    image_dir = []

    for name_test in os.listdir(test_root_path):
        img_name = test_root_path + '/' + name_test
        image_dir.append(img_name)

    return image_dir



def predict(recognizer, test_faces_gray):
    
    predict_result = []
    for index, image_test in enumerate(test_faces_gray):
        result, confidence = recognizer.predict(image_test)
        predict_result.append((result, confidence))

    return predict_result

    

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, size):

    image_result = []

    for index, res in enumerate(predict_results):
        image = cv.imread(test_image_list[index])
        result,confidence = res
        confidence = math.floor(confidence*100)/ 100
        text = train_names[result] + ' ' + str(confidence) + '%'
        x, y, h, w = test_faces_rects[index]
        cv.putText(image, text, (x, y), cv.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 10)
        image_final = cv.resize(image, (size, size))
        image_result.append(image_final)

    return image_result



def combine_and_show_result(image_list, size):
    
    final_image = np.concatenate(image_list, axis = 1)
    cv.imshow('Result', cv.resize(final_image, (size*6, size)))
    cv.waitKey(0)


if __name__ == "__main__":

    train_root_path = 'dataset/train/'

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_face_grays, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    test_root_path = 'dataset/test/'

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_result = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_result, test_image_list, test_faces_rects, train_names, 200)
    combine_and_show_result(predicted_test_image_list, 350)