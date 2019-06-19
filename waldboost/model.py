"""
"""


import numpy as np
from .channels import channel_pyramid
from . import model_pb2
from .training import DStump, DTree
import waldboost


def symbol_name(s):
    return s.__module__ + "." + s.__qualname__


def symbol_from_name(name):
    if name.startswith("builtins"):
        _,name = name.split(".")
    ls = {"numpy":np,"waldboost":waldboost}
    return eval(name, {}, ls)


class Model:
    """
    Classification model
    """
    def __init__(self, shape, channel_opts):
        self.shape = shape
        self.channel_opts = channel_opts
        self.classifier = []

    def channels(self, image):
        """
        Iterator over channel pyramid of the given image
        """
        return channel_pyramid(image, self.channel_opts)

    def scan_channels(self, image):
        return ((chns, scale, self.predict_on_image(chns)) for chns, scale in self.channels(image))

    def get_bbs(self, r, c, scale):
        m,n,_ = self.shape
        return np.atleast_2d( [(c,r,n,m) for r,c in zip(r,c)] ) / scale

    def detect(self, image):
        bbs,scores = [],[]
        for chns, scale in self.channels(image):
            r,c,h = self.predict_on_image(chns)
            bbs.append(self.get_bbs(r,c,scale))
            scores.append(h)
        bbs = np.concatenate( [x for x in bbs if x.size] )
        scores = np.concatenate( [x for x in scores if x.size] )
        return bbs, scores

    def predict(self, X):
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
        return len(self.classifier)

    def __bool__(self):
        return bool(self.classifier)

    def append(self, weak, theta):
        self.classifier.append( (weak, theta) )

    def as_proto(self, proto):
        proto.ClearField("shape")
        proto.ClearField("classifier")
        proto.shape.extend(self.shape)
        proto.channel_opts.shrink = self.channel_opts["shrink"]
        proto.channel_opts.n_per_oct = self.channel_opts["n_per_oct"]
        proto.channel_opts.smooth = self.channel_opts["smooth"]
        proto.channel_opts.target_dtype = symbol_name(self.channel_opts["target_dtype"])
        for f,_ in self.channel_opts["channels"]:
            proto.channel_opts.func.append(symbol_name(f))
        for weak,theta in self.classifier:
            w_pb = proto.classifier.add()
            w_pb.theta = theta
            if isinstance(weak, DTree):
                weak.as_proto(w_pb.dtree)
            if isinstance(weak, DStump):
                weak.as_proto(w_pb.dstump)

    @staticmethod
    def from_proto(proto):
        shape = tuple(proto.shape)
        channel_opts = {
            "shrink": proto.channel_opts.shrink,
            "n_per_oct": proto.channel_opts.n_per_oct,
            "smooth": proto.channel_opts.smooth,
            "target_dtype": symbol_from_name(proto.channel_opts.target_dtype),
            "channels": [ (symbol_from_name(s), ()) for s in proto.channel_opts.func ],
        }
        M = Model(shape, channel_opts)
        for weak_proto in proto.classifier:
            theta = weak_proto.theta
            tp = weak_proto.WhichOneof("weak")
            if tp == "dtree":
                weak = DTree.from_proto(weak_proto.dtree)
            elif tp == "dstump":
                weak = DStump.from_proto(weak_proto.dstump)
            M.append(weak, theta)
        return M

    def save(self, filename):
        proto = model_pb2.Model()
        self.as_proto(proto)
        with open(filename, "wb") as f:
            f.write(proto.SerializeToString())

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            proto = model_pb2.Model()
            proto.ParseFromString(f.read())
            return Model.from_proto(proto)
