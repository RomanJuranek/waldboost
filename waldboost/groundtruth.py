import numpy as np
import bbx
from bbx.boxes import Boxes


class RectFormat:  # TODO Enum
    XYXY = 1  # [xmin, ymin, xmax, ymax]
    XYWH = 0  # [xmin, ymin, width, height]
    YXYX = 2  # [ymin, xmin, ymax, xmax]
    # XXYY
    # YYXX


def bbox_list(rects, format=RectFormat.XYXY, **fields):
    """ Create new BoxList 

    Inputs
    ------
    rects : np.ndarray
        Rects in (N,4) array.
    format : member of RectFormat
        Format of rectangles
    fields : keyword arguments
        Extra fields added to the resulting box list. Expected shape of
        values is (N,...). When the field value shape is (N,) it is converted
        to (N,1) before adding to the list.

    Outputs
    -------
    boxes : BoxList
        New list with fields
    """
    # Check
    if not isinstance(rects, np.ndarray):
        raise ValueError("Rects must be numpy array")
    if not rects.ndim == 2 or rects.shape[1] != 4:
        raise ValueError("Rects must be 2D array with 4 columns")
    # Convert to XYXY
    if format is not RectFormat.XYXY:
        a,b,c,d = np.split(rects, 4, axis=1)
        if format == RectFormat.XYWH:
            rects = np.hstack([a,b,a+c,b+d])
        elif format == RectFormat.YXYX:
            rects = np.hstack([b,a,d,c])

    return bbx.Boxes(rects.astype("f"), **fields)


def read_bbgt(filename):
    """
    Read ground truth from bbGt file.
    See Piotr's Toolbox for details
    """
    boxes = []
    with open(filename,"r") as f:
        signature = f.readline()
        if not signature.startswith("% bbGt version=3"):
            raise ValueError("Wrong file signature")
        rects = []
        ignore = []
        labels = []
        for line in f:
            elms = line.strip().split()
            assert len(elms) == 12, "Invalid file"
            lbl = elms[0]
            rect = tuple(map(float, elms[1:5]))
            ign = int(elms[10])
            rects.append(rect)
            ignore.append(ign)
            labels.append(lbl)
    if not rects:
        rects = np.empty((0,4),"f")
        ignore = np.empty(0,"i")
        labels = np.empty(0,"<U1")
    boxes = bbox_list(np.array(rects,"f"),
                    format=RectFormat.XYWH,
                    ignore=np.array(ignore,"i"),
                    labels=np.array(labels))
    return boxes
