import zipfile
import pickle
import functools
import torch
import warnings
import pathlib


class NotYetLoadedTensor:
    def __init__(self, metatensor, archiveinfo, storageinfo, rebuild_args):
        self.metatensor = metatensor
        self.archiveinfo = archiveinfo
        self.storageinfo = storageinfo
        self.rebuild_args = rebuild_args

    @classmethod
    def rebuild(
        cls,
        storage,
        storage_offset,
        size,
        stride,
        requires_grad,
        backward_hooks,
        metadata=None,
        archiveinfo=None,
    ):
        rebuild_args = (
            storage_offset,
            size,
            stride,
            requires_grad,
            backward_hooks,
            metadata,
        )
        metatensor = torch._utils._rebuild_tensor_v2(
            storage,
            storage_offset,
            size,
            stride,
            requires_grad,
            backward_hooks,
            metadata,
        )
        storageinfo = archiveinfo.cache[storage._cdata]
        return NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)

    def _load_tensor(self):
        # we could / should try to lean heavier on PyTorch's reader
        name, storage_cls, fn, device, size = self.storageinfo
        buffer = self.archiveinfo.zipfile.read(
            str(self.archiveinfo.prefix / "data" / fn)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage = storage_cls.from_buffer(buffer, "native")
        tensor = torch._utils._rebuild_tensor_v2(storage, *self.rebuild_args)
        return tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        loaded_args = [
            (a._load_tensor() if isinstance(a, NotYetLoadedTensor) else a) for a in args
        ]
        res = func(*loaded_args, **kwargs)
        # gc.collect would be costly here, maybe do it optionally
        return res

    def __getattr__(self, name):
        # properties
        ## TODO: device, is_...??
        ## TODO: mH, mT, H, T, data, imag, real
        ## name ???
        if name in {
            "dtype",
            "grad",
            "grad_fn",
            "layout",
            "names",
            "ndim",
            "output_nr",
            "requires_grad",
            "retains_grad",
            "shape",
            "volatile",
        }:
            return getattr(self.metatensor, name)
        if name in {"size"}:
            return getattr(self.metatensor, name)
        raise AttributeError(f"{type(self)} does not have {name}")

    def __repr__(self):
        return f"NotYetLoadedTensor({repr(self.metatensor)})"


class LazyLoadingUnpickler(pickle.Unpickler):
    def __init__(self, file, zipfile, prefix):
        super().__init__(file)
        self.zipfile = zipfile
        self.cache = {}
        self.prefix = prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            res = super().find_class(module, name)
            return functools.partial(NotYetLoadedTensor.rebuild, archiveinfo=self)
        return super().find_class(module, name)

    def persistent_load(self, pid):
        name, cls, fn, device, size = pid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = torch.storage.TypedStorage(dtype=cls().dtype, device="meta")
        self.cache[s._cdata] = pid
        return s


def lazy_load(fn):
    zf = zipfile.ZipFile(fn)
    nl = zf.namelist()
    prefix = pathlib.Path(pathlib.Path(nl[0]).parts[0])
    with zf.open(str(prefix / "data.pkl"), "r") as pkl:
        mup = LazyLoadingUnpickler(pkl, zf, prefix)
        sd = mup.load()
    return sd
