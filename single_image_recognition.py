import cv2
import face_recognition
import matplotlib.pyplot as plt


class SingleImageRecognition:

    @staticmethod
    def identify_image(test_image, test_image_locations, myface_encoding):

        # test_image_locations
        # Getting the encodings of the detected face
        test_image_encoding = face_recognition.face_encodings(test_image)[0]
        # test_image_encoding
        # Comparing the faces in both images (single face image)

        # results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
        results = face_recognition.compare_faces([test_image_encoding], myface_encoding)

        if results[0]:
            print("Match")
            (top, right, bottom, left) = test_image_locations[0]
            cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.startWindowThread()
            cv2.imshow('window', test_image)
            # cv2.imshow('window', cv2.resize(test_image))
            cv2.waitKey(0)
        else:
            print("No Match")
