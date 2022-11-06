from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename

app = Flask(__name__)
 
app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


import faceReg
model = faceReg.Recognition('../FaceClassify_REMASTER_2.h5')
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def main():
    return 'Homepage'
 
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    files = request.files.getlist('files[]')
     
    errors = {}
    success = False
    fileSavedName = []
    for file in files:      
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileSavedName.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'
 
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
    if success:
        res = model.predict(fileSavedName[0])
        resp = jsonify({'message' : 'Files successfully uploaded', 'result' : res})
        resp.status_code = 201
    else:
        resp = jsonify(errors)
        resp.status_code = 500
    return resp
 
if __name__ == '__main__':
    app.run(debug=True)