"""
Detect objects in images

Known issues:
* Only detectors with channels built in the waldboost package can be used since
  custom types need to import propper module
"""


import argparse
import json
import logging
from functools import partial
from multiprocessing import Pool

import cv2

import waldboost as wb


def parse_args():
    parser = argparse.ArgumentParser(description='WaldBoost detector')
    parser.add_argument("-m", "--model", dest='model', type=str, action="append", required=True, help="Models")
    parser.add_argument("files", metavar='files', type=str, nargs='+', help='Images to process')
    # min score
    # separate
    # workers
    return parser.parse_args()


def detect_on_image(filename, model):
    logging.debug(f"Processing {filename}")
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #h,w = image.shape
    #image = cv2.resize(image, (w//1, h//1))
    boxes = wb.detect_multiple(image, *model, separate=False, iou_threshold=0.2, score_threshold=0)
    return {
        "filename": filename,
        "boxes": boxes.get().astype("i").tolist(),
        "scores": boxes.get_field("scores").astype("f").tolist(),
    }


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
            
    models = []

    try:
        for m in args.model:
            logging.info(f"Loading model {m}")
            model = wb.Model.load(m)
            model.channel_opts["n_per_oct"] = 4
            models.append(model)
        logging.info(f"Loaded {len(models)} models")
    except:
        logging.critical("Cannot load models")
        exit(1)

    try:
        logging.info(f"Processing {len(args.files)} images")
        detect = partial(detect_on_image, model=models)
        with Pool(6) as pool:
            logging.info(f"Using pool of 6 workers")
            results = list(pool.imap_unordered(detect, args.files))

    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        exit(1)
    except:
        logging.error("Error occured")
        exit(1)
    
    result_str = json.dumps(results, indent=True)
    print(result_str)

    logging.info("Done")
