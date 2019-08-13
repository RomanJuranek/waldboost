"""
Detect objects in images

Known issues:
* Only detectors with channels built in the waldboost package can be used since
  custom types need to import propper module
"""

import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='WaldBoost detector')
    parser.add_argument("-m", "--model", dest='model', type=str, action="append", required=True, help="Models")
    parser.add_argument("files", metavar='files', type=str, nargs='+', help='Images to process')
    # min score
    # separate
    # sow
    return parser.parse_args()


def detect(image, *models, min_score=0):
    bbs, scores = wb.detect_multiple(image, models)
    mask = scores > min_score
    bbs_nms, scores_nms = bbx.nms(bbs[mask,...], scores[mask], min_group=2, min_overlap=0.2)
    return bbs_nms, scores_nms


def show_image():
    for x,y,w,h in bbs_nms.astype("i"):
        cv2.rectangle(image,(x,y),(x+w,y+h),(64,255,64), 2)
    cv2.imshow("detections", image)
    cv2.waitKey()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    import waldboost as wb
    
    models = []
    for m in args.model:
        logging.info(f"Loading model {m}")
        models.append(wb.Model.load(m))

    import bbx
    import cv2

    for f in args.files:
        logging.info(f"Processing {f}")
        # image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # bbs, scores = detect(image, models, min_score=args.min_score)
        # if args.show:
        #     show_image(image, bbs, scores)
