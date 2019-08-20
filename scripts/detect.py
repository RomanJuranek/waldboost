"""
Detect objects in images

Known issues:
* Only detectors with channels built in the waldboost package can be used since
  custom types need to import propper module
"""


import argparse
import logging
import waldboost as wb
from waldboost.testing import detect
import json
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='WaldBoost detector')
    #parser.add_argument("-m", "--model", dest='model', type=str, action="append", required=True, help="Models")
    #parser.add_argument("files", metavar='files', type=str, nargs='+', help='Images to process')
    # min score
    # separate
    return parser.parse_args()


def detect_on_image(filename, model):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    dt_dict = detect()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

        
    models = []
    for m in args.model:
        logging.info(f"Loading model {m}")
        models.append(wb.Model.load(m))

    from multiprocessing import Pool

    with Pool(4) as p:
        results = p.map(detect_on_image, args.files)

    print(results)
