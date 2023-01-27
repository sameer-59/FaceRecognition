import cv2
import face_recognition
import matplotlib.pyplot as plt


class GroupImageRecognition:

    @staticmethod
    def identify_image(group_test_image, group_test_image_location, myface_encoding):


        for i in group_test_image_location:
            (top, right, bottom, left) = i

            cropped_image = group_test_image[top - 50:bottom + 50, left - 50:right + 50]

            cropped_image_locations = face_recognition.face_locations(cropped_image)  # , model="cnn")
            cropped_image_encoding = face_recognition.face_encodings(cropped_image)

            # print('Cropped Image :')
            # cv2.imshow('group window', cv2.resize(cropped_image, (100, 100)))

            faceDis = face_recognition.face_distance(myface_encoding, cropped_image_encoding)
            print('Face Distance : ', faceDis)

            if faceDis > 0.5:
                print("No Match")
            else:
                print("Match")
                cv2.rectangle(group_test_image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.startWindowThread()
                # cv2_imshow(group_test_image)
                cv2.imshow('group window', cv2.resize(group_test_image, (500, 500)))
                cv2.waitKey(0)


