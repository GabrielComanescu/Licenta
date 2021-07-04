from flask import Flask, render_template, request, jsonify, abort, send_file
import os
from werkzeug.utils import secure_filename
import subprocess
from test import shoot
from test2 import shoot2
import imageio
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/img', methods=['POST'])
def get_image():
    pic = request.files['img']
    if pic:
        pic.save('pu.jpg')
        shoot()

        print(pic)
        return send_file('test\\20_pu.jpg', mimetype='image/jpeg')
    else:
        return 'eroare'
    # return ('ok', 204)

@app.route('/video', methods=['POST'])
def return_image():
    uploaded_files = request.files.getlist('img')
    uploaded_files[0].save('im1.png')
    uploaded_files[1].save('im2.png')
    shoot2()

    saved_images = []
    for filename in os.listdir('output/'):
        saved_images.append(f'output/{filename}')

    saved_images.sort(key=natural_keys)
    saved_images = [imageio.imread(i) for i in saved_images]
    imageio.mimsave('output/output.gif', saved_images)

    return send_file('output/output.gif', mimetype='image/gif')
 


if __name__ == '__main__':
    app.run(debug=True)