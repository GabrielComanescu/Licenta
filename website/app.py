from flask import Flask, render_template, request, jsonify, abort, send_file
import os
from werkzeug.utils import secure_filename
import subprocess
from test import shoot

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

@app.route('/video')
def return_image():
    return 'inca nu avem'


if __name__ == '__main__':
    app.run(debug=True)