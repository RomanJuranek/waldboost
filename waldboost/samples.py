""" Support for generating samples from images
"""

import logging

import numpy as np
import bbx
from bbx import Boxes


def gather_samples(chns, rs, cs, shape):
    """ Crop feature maps

    Input
    -----
    chns : np.ndarray
        x
    rs, cs : np.ndarray
        rows and columns of the samples to crop
    shape : tuple
        Shape of samples

    Output
    ------
    X : np.ndarray
        Cropped samples with shape (rs.size,) + shape + (chns.shape[2],)
        E.g.: When rs (and cs) has size 10 (i.e. 10 samples), shape=(20,20)
        and chns.shape is (100,100,4), the resulting shape will be (10,20,20,4)

    Notes
    -----
    No range checks are performed.
    """
    if rs.size != cs.size:
        raise ValueError("Sizes of 'rs' and 'cs' must match")
    m,n,_ = shape
    if rs.size == 0:
        return np.empty((0,)+shape, dtype=chns.dtype)
    X = [ chns[r:r+m,c:c+n,...] for r,c in zip(rs, cs) ]
    return np.array(X)


def select_candidates(condition, max_candidates:int) -> np.ndarray:
    """ Select at most max_candidates from items where condition evaluates to True
    
    Input
    -----
    condition : ndarray of np.bool
        Array from which candidates are drawn
    max_candidates : int
        Max number of candidates to select from condition==True items

    Output
    ------
    idx : np.ndarray
        List of indices where condition==True. idx.size <= max_candidates.
        Items are selected by np.random.choice() when there are more
        then max_candidate possible items, otherwise np.flatnonzero(condition)
        is returned.

    Notes
    -----
    Order of items in idx is not ensured.

    Example
    -------
    x = np.random.rand(1000)  # 1000 random numbers
    idx = select_candidates(x>0.5, 5)  # Return at most 5 indices of items with value larger than 0.5
    assert np.all(x[idx]>0.5) and idx.size<=5
    """
    idx = np.flatnonzero(condition)
    if idx.size > max_candidates:
        idx = np.random.choice(idx, max_candidates)
    return idx


class SampleLabel:
    """ Constants for labeling samples as true/false positives """
    TRUE_POSITIVE = 1
    FALSE_POSITIVE = -1
    IGNORE = 0


def label_boxes(dt_boxes:Boxes,
                gt_boxes:Boxes,
                min_tp_iou = 0.7,
                max_fp_iou = 0.3,
                max_tp_candidates = 100,
                max_fp_candidates = 100):
    """ Label boxes as TP, FP, or ignore and assign ground truth instance id
    
    Input
    -----
    dt_boxes : BoxList
        List of boxes to be labeled (can contain any extra fields). At most
        max_tp_candidates and max_fp_candidates will be labeled as tp/fp,
        others will be labeled as ignore.
    gt_boxes : BoxList
        List of ground truth boxes with optional 'ignore' field.
    min_tp_iou : float
        Boxes with iou>min_tp_iou are considered as true positives.
    max_fp_iou : float
        Boxes with iou<min_fp_iou are considered as false positives.
    max_tp_candidates : int
        Max number of candidates to select from true positive boxes.
    max_fp_candidates : int
        Max number of candidates to select from false positive boxes.

    Output
    ------
    None, mutates dt_boxes by adding fields 'instance_id' and 'tp_label'

    New fields
    ----------
    tp_label : ndarray
        Each box is labeled by a value from {-1,0,1}
        -1 : false positive box
         0 : ignored box
         1 : true positive box
    instance_id : ndarray
        Index of a box from gt_boxes with highest iou. Valid for true
        positives (tp_label == 1). Values are from range(0, gt_boxes.num_boxes())
    """
    ignore_flag = gt_boxes.get_field("ignore") if gt_boxes.has_field("ignore") else np.zeros(len(gt_boxes))
    if ignore_flag.ndim != 1:
        raise ValueError("'ignore' field must be single dimension")
    if len(gt_boxes) > 0:
        overlap = bbx.iou(dt_boxes, gt_boxes)
        dt_iou = np.max(overlap, axis=1)
        dt_instance_id = np.argmax(overlap, axis=1)
        dt_ignore_flag = ignore_flag[dt_instance_id]
        fp = select_candidates(dt_iou < max_fp_iou, max_fp_candidates)
        tp = select_candidates(np.logical_and(dt_iou > min_tp_iou, dt_ignore_flag == 0), max_tp_candidates)
        box_label = np.full(len(dt_boxes), SampleLabel.IGNORE, np.int32)
        box_label[tp] = SampleLabel.TRUE_POSITIVE
        box_label[fp] = SampleLabel.FALSE_POSITIVE
    else:
        dt_instance_id = np.full(len(dt_boxes), -1, np.int32)
        box_label = np.full(len(dt_boxes), SampleLabel.IGNORE, np.int32)
        fp = select_candidates(np.ones(len(dt_boxes), np.bool), max_fp_candidates)
        box_label[fp] = SampleLabel.FALSE_POSITIVE
    dt_boxes.set_field("instance_id", dt_instance_id)
    dt_boxes.set_field("tp_label", box_label)


def get_regression_target(dt_boxes, gt_boxes):
    if not dt_boxes.has_field("instance_id"):
        raise ValueError("'instance_id' field is missing")
    gt_idx = dt_boxes.get_field("instance_id")
    regression_target = dt_boxes.get() - gt_boxes[gt_idx].get()
    dt_boxes.add_field("regression_target", regression_target)


def get_samples_from_image(model,
                           image,
                           gt_boxes,
                           tp = True,
                           fp = True,
                           **kwargs):
    """ Get samples from image

    Input
    -----
    model : wb.Model
        Detection model - a producer of feature maps and locations of objects
    image : np.ndarray
        Grayscale image
    gt_boxes : BBoxList
        List of ground truth bounding boxes with optional 'ignore field'
    tp/fp : bool, int
        Include tp/fp when evaluates to True
    kws :
        Additional arguments form label_boxes(). Useful for setting
        iou thresholds for tp and fp.

    Output
    ------
    dt_boxes : BBoxList
        List of bounding boxes with fields `scores`, 'tp_label', and 'samples'.
        'scores' represent classifier response value. 'tp_label' is 1 for
        true positives, 1 for false positives. 'samples' contain corresponding
        feature maps form the image.

    See also
    --------
    label_boxes : Function that decides what is tp/fp/ignored
    """
    box_lists = []
    for chns,scale,(r,c,h) in model.scan_channels(image):
        # Get dt_boxes from locations detected by model
        if r.size == 0: continue
        dt_boxes = Boxes(model.get_boxes(r,c,scale))  # dt_boxes in the original image space
        dt_boxes.set_field("scores", h)
        dt_boxes.set_field("row", r)
        dt_boxes.set_field("col", c)
        #print(chns.shape)
        # Label the detections
        label_boxes(dt_boxes, gt_boxes, **kwargs)
        # Select what should be included in output
        tp_label = dt_boxes.get_field("tp_label")
        sample_selector = np.logical_or(
            np.logical_and(tp_label== 1, tp),
            np.logical_and(tp_label==-1, fp))
        sample_indices = np.reshape(np.where(sample_selector), [-1])
        dt_boxes = dt_boxes[sample_indices]
        # Crop the feature maps
        samples = gather_samples(chns, dt_boxes.get_field("row").flatten(), dt_boxes.get_field("col").flatten(), model.shape)
        dt_boxes.set_field("samples", samples)
        box_lists.append(dt_boxes)
    return bbx.concatenate(box_lists)


class SamplePool(object):
    """ Container for training samples """
    def __init__(self,
                 min_tp=1000,
                 min_fp=1000,
                 logger=None,
                 **kwargs):
        """
        Inputs
        ------
        min_tp, min_fp : int
            Minimal number of training samples to keep
        logger : Logger
            Optional logger instance
        kwargs :
            Additional parameters passed to get_samples_from_image()

        Example
        -------
        pool = SamplePool()
        pool.update(model, [(image, boxes)])
        X,H = pool.get_true_positives()
        """
        self.samples = []
        self.min_tp = min_tp
        self.min_fp = min_fp
        self.n_tp = 0
        self.n_fp = 0
        self.label_boxes_args = kwargs
        self.logger = logger or logging.getLogger("SamplePool")

    def print_stats(self):
        print("Pool stats:")
        print(f"tp: {self.n_tp}; fp: {self.n_fp}")
        print(f"Require tp: {self.require_tp}; fp: {self.require_fp}")
        pass

    def update(self, model, gen):
        self.prune(model)
        if self.require_tp or self.require_fp:
            #for image,gt_boxes,*_ in gen:
            for gt_dict in gen:
                self.print_stats()
                image = gt_dict.get("image")
                gt_boxes = gt_dict.get("groundtruth_boxes")
                dt_boxes = get_samples_from_image(model, image, gt_boxes, tp=self.require_tp, fp=self.require_fp, **self.label_boxes_args)
                if len(dt_boxes) == 0:
                    continue
                self.samples.append(dt_boxes)
                sample_label = dt_boxes.get_field("tp_label")
                new_tp = (sample_label== 1).sum()
                new_fp = (sample_label==-1).sum()
                self.n_tp += new_tp
                self.n_fp += new_fp
                self.logger.debug(f"Added {new_tp} tp and {new_fp} fp samples")
                if not self.require_tp and not self.require_fp:
                    break
            else:
                self.logger.debug("Iterator exhausted")
        self.print_stats()

    @property
    def require_tp(self):
        """ Number of tp samples required to fill the pool """
        return max(0, self.min_tp - self.n_tp)

    @property
    def require_fp(self):
        """ Number of fp samples required to fill the pool """
        return max(0, self.min_fp - self.n_fp)

    def prune(self, model):
        """ Remove samples rejected by the model from the pool """
        pruned_samples = []
        n_tp = 0
        n_fp = 0
        for s in self.samples:
            score,mask = model.predict(s.get_field("samples"))
            keep_indices = np.flatnonzero(mask)
            if keep_indices.size == 0:
               continue
            score_field = s.get_field("scores")
            score_field[:] = score  # FIXME this is a hack
            pruned_s = s if keep_indices.size==len(s) else s[keep_indices]
            pruned_samples.append(pruned_s)
            sample_label = pruned_s.get_field("tp_label")
            n_tp += (sample_label== 1).sum()
            n_fp += (sample_label==-1).sum()
        self.samples = pruned_samples  # concatenate ?
        self.n_tp = n_tp
        self.n_fp = n_fp
        self.print_stats()
    
    def filter_by_tp_label(self, label):
        boxes = []
        for s in self.samples:
            mask = s.get_field("tp_label") == label
            mask = np.flatnonzero(mask)
            if mask.size > 0:
                #print(mask)
                boxes.append(s[mask])
        #print([b.num_boxes() for b in boxes])
        return bbx.concatenate(boxes)

    def get_samples(self, label):
        boxes = self.filter_by_tp_label(label=label)
        X = boxes.get_field("samples")
        H = boxes.get_field("scores").flatten()
        return X, H

    def get_true_positives(self):
        """
        Return true positive samples.

        Output
        ------
        X : ndarray
            Sample feature maps with shape (N,H,W,C)
        H : ndarray
            Sample scores with shape (N,)
        """
        return self.get_samples(label=1)

    def get_false_positives(self):
        """
        Return false positive samples.

        Output
        ------
        X : ndarray
            Sample feature maps with shape (N,H,W,C)
        H : ndarray
            Sample scores with shape (N,)
        """
        return self.get_samples(label=-1)
