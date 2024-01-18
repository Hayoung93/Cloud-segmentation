import os
import sys

from flask import Flask, render_template, request, send_file
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from werkzeug.utils import secure_filename
from datetime import datetime

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from resnet50_unetpp import UNetWithResnet50Encoder

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def single_infer(file_path):
    img = Image.open(file_path).convert("RGB")
    img = resize(img)
    img = normalize(totensor(img)).unsqueeze(0).cuda().to(torch.float32)
    with torch.inference_mode():
        out = model(img)
        out = torch.sigmoid(out[0][0]).data.cpu().numpy()
    out_ = (out > 0.5).astype(np.uint8) * 255
    out_name = ".".join(file_path.split("/")[-1].split(".")[:-1]) + "_result.png"
    out_fp = os.path.join(app.config["UPLOAD_DIR"], out_name)
    cv2.imwrite(out_fp, out_)
    return out_fp



def create_app():
    application = Flask(__name__)
    Bootstrap(application)
    navigation = Nav(application)

    return application, navigation

app, nav = create_app()
app.config['UPLOAD_DIR'] = "./static"


@app.route('/demo/seg_api', methods=['POST'])
def captioning_api():
    if request.method == 'POST':
        f = request.files['input_image']
        filename = secure_filename(f.filename)
        filename = os.path.join(app.config['UPLOAD_DIR'], filename)
        fname, ext = os.path.splitext(filename)
        time_string = datetime.now().strftime('_%Y-%m-%d_%H:%M:%S')
        filename_save = fname + time_string + ext
        f.save(filename_save)
        print(filename_save)

        output_name = single_infer(filename_save)

        return send_file(output_name, mimetype="image/png")



@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    os.makedirs("./static", exist_ok=True)
    model = UNetWithResnet50Encoder(1)
    cp = torch.load("./model.pth")
    model.load_state_dict(cp)
    model = model.eval().cuda()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    totensor = transforms.ToTensor()
    resize = transforms.Resize((384, 384))

    app.run(host='0.0.0.0', port=22334, debug=True)
