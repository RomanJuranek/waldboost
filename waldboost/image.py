import numpy as np
import math
import cv2


def normalize(image):
    low, high = np.percentile(image, [1,99])
    return (image - low) / (high - low)


def noise(image, level):
    u,v = image.shape
    return image + level * np.random.randn(u,v)


def contrast(image):
    low = 0.05 * np.random.randn() + 0
    high = 0.05 * np.random.randn() + 1
    return (image - low) * (high - low)


def disk_kernel(radius=1):
    radius = max(radius, 1)
    sz = float(2 * math.ceil(radius) + 1)
    x, y = np.ogrid[1:sz, 1:sz]
    H = (x - sz / 2) ** 2 + (y - sz / 2) ** 2 < radius ** 2
    return (H / np.sum(H)).astype(np.float)


def defocus_blur(image, size):
    H = disk_kernel(size)
    return cv2.filter2D(image, cv2.CV_32F, H)


def motion_blur(image, angle, length=2):
    pass


def gamma(image, g):
    return image ** (1/g)


def random_adjust(image):
    image = image.astype(np.float32) / 256

    # normalize
    #image = normalize(image)

    image = gamma(image, 1*np.random.rand()+0.5)

    # blur
    image = defocus_blur(image, np.random.rand()*2)

    # noise
    image = noise(image, np.random.rand()*0.02)

    # brightness, contrast
    image = contrast(image)

    image = (np.clip(image*255,0,255)).astype(np.uint8)

    #_,b = cv2.imencode(".jpg",image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    #image = cv2.imdecode(b,cv2.IMREAD_GRAYSCALE)

    return image
