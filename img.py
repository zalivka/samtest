import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from flask import Flask, request, render_template
import matplotlib.pyplot as plt
# import vtracer
from sam2.build_sam import build_sam2



app = Flask(__name__)

from cut2 import click_handler_bp

app.register_blueprint(click_handler_bp)


@app.route('/')
def index():
    # vtracer.convert_image_to_svg_py(image_path='static/mask_0.png', out_path='static/mask_0_vectorized.svg')
    return render_template('test.html')

@app.route('/success_page')
def success():
    import time
    
    start_time = time.time()
    # Your existing code for the success route goes here
    end_time = time.time()
    
    time_elapsed = end_time - start_time
    print(f"Time elapsed: {time_elapsed:.4f} seconds")
    return "???"
    # return render_template('res.html')

import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return process_uploaded_file(file_path)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def process_uploaded_file(file_path):
    # Get the filename from the file path
    filename = os.path.basename(file_path)
    
    # Render a template that displays the uploaded image
    return render_template('uploaded_image.html', filename=filename)


# @app.route('/get-coordinates', methods=['GET'])
# def get_coordinates():
#     x = request.args.get('x')
#     y = request.args.get('y')
#     print(f'Clicked coordinates: x={x}, y={y}')
#     return 'Coordinates received'

# if __name__ == '__main__':
#     app.run()

if __name__ == '__main__':
    app.run(host='0.0.0.0')