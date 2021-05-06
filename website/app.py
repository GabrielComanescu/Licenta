from flask import Flask, render_template, request, jsonify, abort
import os
from werkzeug.utils import secure_filename
import subprocess
from test import shoot

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get_image():
    pic = request.files['img']
    pic.save('pu.jpg')
    shoot()

    print(pic)
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)