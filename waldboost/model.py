""" Representation of detection model

See also
--------
waldboost.model.Model
"""


import numpy as np
from .channels import channel_pyramid
from . import model_pb2
from .training import DTree
import waldboost
import zlib
from google.protobuf.message import DecodeError


def symbol_name(s):
    return s.__module__ + "." + s.__qualname__


def symbol_from_name(name, custom_objects={}):
    from importlib import import_module
    m,rest = name.split(".",1)
    ls = {m: import_module(m)}
    return eval(name, {}, ls)


class Model:
    """ Detection model

    The model holds 1/ size of detection window (shape), 2/ channel calculation
    parameters (channel_opts), and 3/ the classifier (sequence of weak classifiers)

    New empty instance is initialized as:
        model = Model(shape, channel_opts)

    The model can be used in two distinct modes:
    - classify individual samples - model.predict(X). This is used during training
      for classification of training data. X must be np.array with shape (N,)+model.shape.

    - classify sub-windows of channel image - model.predict_on_image(X). Used during
      detection for classification of channel images generated from input image.
      In this case shape of X must be (H,W,C) where H, W are arbitrary sizes and
      C=model.shape[-1].

    The full detection is encapsulated in Model.detect(image) which returns list
    of detected bounding boxes and scores. The image must be whatever accepted
    by waldboost.channels.channel_pyramid(), usually np.array with shape (H,W,1)
    and uint8 dtype.

        bbs, scores = model.detect(image)

    Model can be stored in binary and restored from it again.

        model.save("model.pb")
        M = waldboost.load("model.pb")
    """
    def __init__(self, shape, channel_opts):
        self.shape = shape
        self.channel_opts = channel_opts
        self.classifier = []

    def channels(self, image):
        """ Iterator over channel pyramid of the given image

        Example:
        for chns, scale in model.channels():
            # chns is (H,W,C) array with channel features extracted by channel_pyramid
            # scale - coordinates r,c in chns correspond to r/scale, c/scale in image
        """
        return channel_pyramid(image, self.channel_opts)

    def scan_channels(self, image):
        """ Generator of channels and locations detected on gather_samples

        Inputs
        ------
        image : ndarray
            Input image with dimensions (H,W,C) where W, H are arbitrary sizes
            and C is model.shape[-1]

        Outputs
        -------
        g : generator
            Generator that can be used as chns,scale,locations = next(g), where
            chns, scale correspond to output of model.channels() and locations
            are produced by model.predict_on_image.

        Notes
        -----
        This method is useful for generating feature channel images and corresponding
        locations that pass the classifier in the model. Internally used by
        samples.Pool class.

        Example
        -------
        for chns, scale, (r,c,h) in model.scan_channels(image):
            # chns, scale correspond to output of model.channels()
            # r,c - detected locations chns[r:r+h,c:c+w,:] where where h,w,c = model.shape
            # h - detector scores
        """
        return ((chns, scale, self.predict_on_image(chns)) for chns, scale in self.channels(image))

    def get_bbs(self, r, c, scale):
        m,n,_ = self.shape
        return np.atleast_2d( [(c,r,n,m) for r,c in zip(r,c)] ) / scale

    def detect(self, image):
        """ Detect objects in image

        Parameters
        ----------
        image : ndarray
            Input image with dimensions (H,W,C) where W, H are arbitrary sizes
            and C is model.shape[-1]

        Returns
        -------
        bbs : ndarray
            Bounding boxes of deteced objects with shape (N,4) and dtype
            np.float32. Each item in bbs corresponsds to bounding box location
            in x,y,w,h format.

        scores : ndarray
            Classifier response value for each bounding box (float32)

        Examples
        --------
        model = waldboost.load_model("model.pb")
        bbs, scores = model.detect(image)

        for (x,y,w,h), h iqn zip(bbs, scores):
            # Process bbox
        """
        bbs,scores = [],[]
        for chns, scale in self.channels(image):
            r,c,h = self.predict_on_image(chns)
            bbs.append(self.get_bbs(r,c,scale))
            scores.append(h)
        bbs = [x for x in bbs if x.size]
        scores = [x for x in scores if x.size]
        if bbs:
            bbs = np.concatenate(bbs)
            scores = np.concatenate(scores)
        return np.array(bbs), np.array(scores)

    def predict(self, X):
        """ Predict model on samples

        Inputs
        ------
        X : ndarray
            Input samples with shape (N,) + model.shape

        Outputs
        -------
        H : ndarray
            Responses of model on samples
        mask : ndarray
            Flag for each sample (True is sample passed the classifier)

        Notes
        -----

        Example
        -------

        X,_ = pool.gather_samples(0)  # Take background samples from pool
        H,mask = model.predict(X)

        """
        n,*shape = X.shape
        assert tuple(shape) == tuple(self.shape), f"Invalid shape of X. Expected {self.shape}, given {shape}"
        H = np.zeros(n, np.float32)
        mask = np.ones(n, np.bool)
        for weak, theta in self.classifier:
            H[mask] += weak.predict(X[mask,...])
            if theta == -np.inf:
                continue
            mask = np.logical_and(mask, H>=theta)
        return H, mask

    def predict_on_image(self, X):
        """ Predict the model on all positions of image

        Inputs
        ------
        X : ndarray

        Outputs
        -------
        r,c : ndarray
            rows and columns of location that passed the classifier
        h : ndarray
            Classifier response for each location

        Example
        -------

        for chns, scale in model.channels():
            r,c,h = model.predict_on_image(chns)
        """
        u,v,ch_image = X.shape
        m,n,ch_cls = self.shape
        assert ch_image == ch_cls, f"Invalid number of channels. Expected {ch_cls} given {ch_image}."
        idx = np.arange(max(u-m,0)*max(v-n,0), dtype=np.int32)
        rs = idx % (u-m)
        cs = idx // (u-m)
        hs = np.zeros_like(rs, np.float32)
        for weak, theta in self.classifier:
            hs += weak.predict_on_image(X, rs, cs)
            if theta == -np.inf:
                continue
            mask = hs >= theta
            rs = rs[mask]
            cs = cs[mask]
            hs = hs[mask]
        return rs, cs, hs

    def __len__(self):
        """ Length of the classifier """
        return len(self.classifier)

    def __bool__(self):
        """ True if there is at least one stage in the classifier """
        return bool(self.classifier)

    def append(self, weak, theta):
        """ Add new stage to the classifier

        Inputs
        ------
        weak : training.DStump or training.DTree or whatever with the same interface
            The weak classifier.
        theta : scalar
            Rejection threshold
        """
        self.classifier.append( (weak, theta) )

    def as_proto(self, proto):
        """ Fill model_pb2.Model structure

        Inputs
        ------
        proto : model_pb2.Model
            Uninitialized structure to be filled

        Notes
        -----
        Fills the structure or overwrites whatever there was.
        """
        proto.ClearField("shape")
        proto.ClearField("classifier")
        proto.shape.extend(self.shape)
        proto.channel_opts.shrink = self.channel_opts["shrink"]
        proto.channel_opts.n_per_oct = self.channel_opts["n_per_oct"]
        proto.channel_opts.smooth = self.channel_opts["smooth"]
        proto.channel_opts.target_dtype = symbol_name(self.channel_opts["target_dtype"])
        for f in self.channel_opts["channels"]:
            proto.channel_opts.func.append(symbol_name(f))
        for weak,theta in self.classifier:
            w_pb = proto.classifier.add()
            w_pb.theta = theta
            # if isinstance(weak, DTree):
            #     weak.as_proto(w_pb.dtree)
            # if isinstance(weak, DStump):
            #     weak.as_proto(w_pb.dstump)
            # #if isinstance(weak, SKLearnDTree):
            weak.as_proto(w_pb.dtree1)

    @staticmethod
    def from_proto(proto, custom_objects={}):
        """ Create new instance from model_pb2.Model """
        shape = tuple(proto.shape)
        channel_opts = {
            "shrink": proto.channel_opts.shrink,
            "n_per_oct": proto.channel_opts.n_per_oct,
            "smooth": proto.channel_opts.smooth,
            "target_dtype": symbol_from_name(proto.channel_opts.target_dtype, custom_objects),
            "channels": [ symbol_from_name(s, custom_objects) for s in proto.channel_opts.func ],
        }
        M = Model(shape, channel_opts)
        for weak_proto in proto.classifier:
            theta = weak_proto.theta
            weak = DTree.from_proto(weak_proto.dtree1)
            #tp = weak_proto.WhichOneof("weak")
            # print(tp)
            # if tp == "dtree":
            #     weak = DTree.from_proto(weak_proto.dtree)
            # elif tp == "dstump":
            #     weak = DStump.from_proto(weak_proto.dstump)
            # elif tp == "dtree1":
            #     weak = SKLearnDTree.from_proto(weak_proto.dtree1)
            # else:
            #     print(f"Unknown type {tp}")
            M.append(weak, theta)
        return M

    def save(self, filename, compress=False):
        """ Save model to protobuf file """
        proto = model_pb2.Model()
        self.as_proto(proto)
        data = proto.SerializeToString()
        if compress:
            data = zlib.compress(data, 9)
        with open(filename, "wb") as f:
            f.write(data)

    @staticmethod
    def load(filename, custom_objects={}):
        """ Load model from protobuf file and return new instance """
        with open(filename, "rb") as f:
            data = f.read()
        proto = model_pb2.Model()
        try:
            proto.ParseFromString(data)
        except DecodeError:
            data = zlib.decompress(data)
            proto.ParseFromString(data)
        return Model.from_proto(proto, custom_objects)


# from numba import jit
#
# class DTreeModel:
#     def __init__(model):
#         self.model = model;
#         self.T = len(self.model)
#         self.n = 3
#         self.ftr = np.empty((T,n,3), "i")
#         self.thr = np.empty((T,n), "f")
#         self.h = np.empty((T,n,2), "f")
#         self.theta = np.empty(t, "f")
#         self.leaf = np.empty((T,n), "b")
#         for t, (weak, theta) in enumerate(self.model.classifier):
#             assert(isinstance(weak, DTree))
#             self.theta[t] = theta
#             for j in range(n):
#                 self.ftr[t,j,:] = weak.nodes[j].ftr
#                 self.thr[t,j] = weak.nodes[j].thr
#                 self.h[t,j,:] = weak.nodes[j].h
#                 self.leaf[]
#
#     @jit
#     def predict_on_image(X):
#         h,w = x.shape[:2]
#         res = []
#         for y in range(h):
#             for x in range(w):
#                 h = 0
#                 for t in range(self.T): # weak loop
#                     node = 0
#                     while self.leaf[t,node]:
#                         r,c,ch = self.ftr[t, node]
#                         bin = X[y+r,x+c,ch] > self.thr[node]
#                         node = 2*node + bin
#                     h += self.h[t,node]
#                     if h < self.theta[t]:
#                         break
#                 else:
#                     res.append( (y,x,h) )
#         return res