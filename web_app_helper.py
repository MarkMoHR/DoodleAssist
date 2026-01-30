import urllib
import cv2
import numpy as np
import base64


# HELPER FUNCTIONS
# helper function to prevent out of range error
def soft(x, lower, upper):
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x


# helper function to convert URL to actual image that is read by opencv
# if channel = 1, black-white image; if channel = 3, rgb image; otherwise, leave it as it is
def url2image(url, channel=None):
    if url == "":
        return None
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if channel == 3:  # by default read the images as colored images
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    elif (
        channel == 1
    ):  # if optional argument color is given as False, then read as black-and-white image
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


def image2url(image):
    _, buffer = cv2.imencode(".png", image)
    image_url = base64.b64encode(buffer).decode("utf-8")
    return "data:image/png;base64," + image_url