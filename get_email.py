from google.cloud import firestore
from google.oauth2 import service_account
import numpy as np
import face_recognition
import cv2
import requests
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image 
from flask_cors import CORS






credentials = service_account.Credentials.from_service_account_file(
    'finalseniorproject-83a2c-firebase-adminsdk-vw35h-324c0b1357.json'
)
firebase = firestore.Client(credentials=credentials)
collection_ref = firebase.collection('faces')

def get_email(img):

    try:
        rgb_image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = face_recognition.face_encodings(rgb_image, face_recognition.face_locations(rgb_image) )[0]
        docs = collection_ref.stream()
        probs=[]
        email=[]


        for doc in docs:
            dic=doc.to_dict()
            img2=np.fromstring(dic['face'], dtype=float, sep=" ")

            if face_recognition.compare_faces([img1], img2):
                face_distance = face_recognition.face_distance([img1], img2)
                probs.append(1 - face_distance[0])
                email.append(dic['email'])
        
        api_key = 'AIzaSyBst-EgWuwLNQE9mOXqdIvXZjb2sBTiXdQ'
        url = f'https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}'
        payload = {
        'requestType': 'PASSWORD_RESET',
        'email': email[probs.index(max(probs))]}
        response = requests.post(url, json=payload)

        if response.status_code==200:
            return True
        
    except IndexError:
        return False

app = Flask(__name__)
CORS(app)  

@app.route('/get_email', methods=['POST'])
def get_email_endpoint():


    data = request.get_json()
    image_data = data['image']
    image_bytes = base64.b64decode(image_data)
    image =np.array(  Image.open(BytesIO(image_bytes)) )

    mes=get_email(image)
    s=''
    if mes:
        s='1'
    else:
        s='0'

    return jsonify({'message': s}), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False)