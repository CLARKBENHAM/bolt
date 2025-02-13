# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_bolt', [dirname(__file__)])
        except ImportError:
            import _bolt
            return _bolt
        if fp is not None:
            try:
                _mod = imp.load_module('_bolt', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _bolt = swig_import_helper()
    del swig_import_helper
else:
    import _bolt
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


class BoltEncoder(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoltEncoder, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BoltEncoder, name)
    __repr__ = _swig_repr

    def __init__(self, nbytes, scaleby=1.0):
        this = _bolt.new_BoltEncoder(nbytes, scaleby)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _bolt.delete_BoltEncoder
    __del__ = lambda self: None

    def set_centroids(self, *args):
        return _bolt.BoltEncoder_set_centroids(self, *args)

    def set_data(self, *args):
        return _bolt.BoltEncoder_set_data(self, *args)

    def set_offsets(self, v):
        return _bolt.BoltEncoder_set_offsets(self, v)

    def set_scale(self, a):
        return _bolt.BoltEncoder_set_scale(self, a)

    def dists_sq(self, q):
        return _bolt.BoltEncoder_dists_sq(self, q)

    def dot_prods(self, q):
        return _bolt.BoltEncoder_dot_prods(self, q)

    def knn_l2(self, q, k):
        return _bolt.BoltEncoder_knn_l2(self, q, k)

    def knn_mips(self, q, k):
        return _bolt.BoltEncoder_knn_mips(self, q, k)

    def set_codes(self, *args):
        return _bolt.BoltEncoder_set_codes(self, *args)

    def centroids(self):
        return _bolt.BoltEncoder_centroids(self)

    def codes(self):
        return _bolt.BoltEncoder_codes(self)

    def lut_l2(self, *args):
        return _bolt.BoltEncoder_lut_l2(self, *args)

    def lut_dot(self, *args):
        return _bolt.BoltEncoder_lut_dot(self, *args)

    def get_lut(self):
        return _bolt.BoltEncoder_get_lut(self)

    def get_offsets(self):
        return _bolt.BoltEncoder_get_offsets(self)

    def get_scale(self):
        return _bolt.BoltEncoder_get_scale(self)
BoltEncoder_swigregister = _bolt.BoltEncoder_swigregister
BoltEncoder_swigregister(BoltEncoder)

# This file is compatible with both classic and new-style classes.


