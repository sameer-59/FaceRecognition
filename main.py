from flask import Flask
from flask_restx import Api, Resource, reqparse
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.datastructures import FileStorage
import face_recognition

from group_image_recognition import GroupImageRecognition
from single_image_recognition import SingleImageRecognition

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='Face Recognition API', description='A simple API', )
app.config['Upload_folder'] = './images/'

ns = api.namespace('FaceRecognition', description='Api to upload images')

file_upload = reqparse.RequestParser()
file_upload.add_argument('image',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='upload image')


@ns.route('/')
class FaceRecognition(Resource):

    @ns.doc('upload image')
    @ns.expect(file_upload)
    def post(self):
        ''' Upload Image '''
        input_data = file_upload.parse_args()
        user_image = input_data['image']

        my_image = face_recognition.load_image_file(user_image)

        myface_encoding = face_recognition.face_encodings(my_image)[0]

        test_image = face_recognition.load_image_file("./images/1674710570330.jpg")

        # CNN?
        test_image_landmarks = face_recognition.face_landmarks(test_image)

        if len(test_image_landmarks) > 1:
            GroupImageRecognition.identify_image(group_test_image=test_image, myface_encoding=myface_encoding)
        else:
            SingleImageRecognition.identify_image(test_image=test_image, myface_encoding=myface_encoding)

        return {"message": "Hello"}, 201


if __name__ == '__main__':
    app.run(debug=True)
