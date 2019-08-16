import numpy as np

from . import bbox


class AspectRatio:
    KEEP_WIDTH = 0
    KEEP_HEIGHT = 1
    EXPAND = 2
    SHRINK = 3
    KEEP_AREA = 4


def set_aspect_ratio(boxes, ar=1, action=AspectRatio.KEEP_WIDTH):
    """ Set aspect ratio of boxes without moving center

    Inputs
    ------
    boxes : BoxList
        Boxes whose aspect ration will be changed
    ar : float
        Desired aspect ratio (width/height)
    action : member of AspectRatio
        The action determinig aspect ratio adjustment strategy

    Outputs
    -------
    new_boxes : BoxList
        Boxes corresponding to input boxes with the same fields
    """
    if ar <= 0:
        raise ValueError("Aspect ratio must be positive float")

    rects = boxes.get()
    width = rects[:,3] - rects[:,1]
    height = rects[:,2] - rects[:,0]

    # Calculate new width and height according to selected strategy
    if action is AspectRatio.KEEP_WIDTH:
        new_width = width
        new_height = width / ar
    elif action is AspectRatio.KEEP_HEIGHT:
        new_width = height * ar
        new_height = height
    elif action is AspectRatio.EXPAND or action is AspectRatio.SHRINK:
        aspect_ratio = width / height
        condition = aspect_ratio > ar if action is AspectRatio.EXPAND else aspect_ratio < ar
        new_width = np.where(condition, width, height*ar)
        new_height = np.where(condition, width/ar, height)
    elif action is AspectRatio.KEEP_AREA:
        area = width * height
        new_width = area * ar
        new_height = area / new_width
    else:
        raise ValueError("Unknown action")

    # Calculate change for each coordinate
    width_change = (new_width - width) / 2
    height_change = (new_height - height) / 2
    change = np.array([-height_change, -width_change, height_change, width_change]).transpose()
    new_rects = rects + change

    # Init new list and copy the fields
    new_boxes = bbox.BoxList(new_rects)
    bbox.np_box_list_ops._copy_extra_fields(new_boxes, boxes)

    return new_boxes


class RectFormat:
    XYWH = 0  # [xmin, ymin, width, height]
    XYXY = 1  # [xmin, ymin, xmax, ymax]
    YXYX = 2  # [ymin, xmin, ymax, xmax]


def bbox_list(rects, format=RectFormat.YXYX, **fields):
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
    # Convert to YXYX
    if format is not RectFormat.YXYX:
        a,b,c,d = np.split(rects, 4, axis=1)  # pylint: disable=unbalanced-tuple-unpacking
        if format == RectFormat.XYWH:
            rects = np.hstack([b,a,b+d-1,a+c-1])
        elif format == RectFormat.XYXY:
            rects = np.hstack([b,a,d,c])
    boxes = bbox.BoxList(rects.astype("f"))
    n = boxes.num_boxes()
    for field_name, value in fields.items():
        if value.size==n and value.ndim==1:
            value = value[...,None]
        boxes.add_field(field_name, value)
    return boxes


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
    boxes = None
    if rects:
        boxes = bbox_list(np.array(rects,"f"),
                          format=RectFormat.XYWH,
                          ignore=np.array(ignore),
                          labels=np.array(labels))
    return boxes
