import os
import requests
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="Path to input image")
    parser.add_argument("--address", type=str, default="http://0.0.0.0:22334/demo/seg_api")
    parser.add_argument("--output_file", type=str, help="Output file path to save recieved image")
    args = parser.parse_args()
    return args


def main(args):
    # check image
    assert os.path.isfile(args.input_image)
    ext = args.input_image.split(".")[-1]
    assert ext in ["jpg", "png"], "Input image file format must be jpg or png (lowercase)"

    # data to send
    data = {"input_image": (args.input_image, open(args.input_image, "rb")), "Content-Type": "image/{}".format(ext)}

    # send & recieve data
    response = requests.post(args.address, files=data)

    # write recieved image
    with open(args.output_file, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    args = get_args()
    main(args)
