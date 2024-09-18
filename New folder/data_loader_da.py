import monai
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.data import Dataset, ArrayDataset, DataLoader
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld,\
                             RandAxisFlipd, RandGaussianNoised, RandGibbsNoised, RandSpatialCropd, \
                             CropForegroundd,AdjustContrastd)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import (  # noqa: F401
    LazyTransform,
    MapTransform,
    Randomizable,
    RandomizableTransform,
    Transform,
    apply_transform,
)
import pandas as pd
import numpy as np
from monai.data.utils import pad_list_data_collate
from typing import Optional

import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from monai.config import NdarrayOrTensor
from monai.transforms.traits import ThreadUnsafe

from monai.apps.utils import get_logger
from monai.config import NdarrayOrTensor

# For backwards compatibility (so this still works: from monai.transforms.compose import MapTransform)
from monai.transforms.lazy.functional import apply_pending_transforms
from monai.transforms.traits import ThreadUnsafe
from monai.utils import MAX_SEED, TraceKeys, TraceStatusKeys, ensure_tuple, get_seed


def execute_compose(
    data: [NdarrayOrTensor , Sequence[NdarrayOrTensor] , Mapping[Any, NdarrayOrTensor]],
    transforms: Sequence[Any],
    map_items: bool = True,
    unpack_items: bool = False,
    start: int = 0,
    end: [int , None] = None,
    lazy: [bool , None] = False,
    overrides: [dict , None] = None,
    threading: bool = False,
    log_stats: [bool , str] = False,
) -> [NdarrayOrTensor , Sequence[NdarrayOrTensor] , Mapping[Any, NdarrayOrTensor]]:
    """
    `execute_compose` provides the implementation that the `Compose` class uses to execute a sequence
    of transforms. As well as being used by Compose, it can be used by subclasses of
    Compose and by code that doesn't have a Compose instance but needs to execute a
    sequence of transforms is if it were executed by Compose. It should only be used directly
    when it is not possible to use `Compose.__call__` to achieve the same goal.
    Args:
        data: a tensor-like object to be transformed
        transforms: a sequence of transforms to be carried out
        map_items: whether to apply transform to each item in the input data if data is a list or tuple.
            defaults to True.
        unpack_items: whether to unpack input data with * as parameters for the callable function of transform.
            defaults to False.
        start: the index of the first transform to be executed. If not set, this defaults to 0
        end: the index after the last transform to be executed. If set, the transform at index-1
            is the last transform that is executed. If this is not set, it defaults to len(transforms)
        lazy: whether to enable :ref:lazy evaluation<lazy_resampling> for lazy transforms. If False, transforms will be
            carried out on a transform by transform basis. If True, all lazy transforms will
            be executed by accumulating changes and resampling as few times as possible.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when
            :ref:lazy evaluation<lazy_resampling> is enabled for the pipeline or a given transform. If lazy is False
            they are ignored. Currently supported args are:
            {`"mode"`, `"padding_mode"`, `"dtype"`, `"align_corners"`, `"resample_mode"`, `device`}.
        threading: whether executing is happening in a threaded environment. If set, copies are made
            of transforms that have the `RandomizedTrait` interface.
        log_stats: this optional parameter allows you to specify a logger by name for logging of pipeline execution.
            Setting this to False disables logging. Setting it to True enables logging to the default loggers.
            Setting a string overrides the logger name to which logging is performed.

    Returns:
        A tensorlike, sequence of tensorlikes or dict of tensorlists containing the result of running
        data` through the sequence of `transforms`.
    """
    end_ = len(transforms) if end is None else end
    if start is None:
        raise ValueError(f"'start' ({start}) cannot be None")
    if start < 0:
        raise ValueError(f"'start' ({start}) cannot be less than 0")
    if start > end_:
        raise ValueError(f"'start' ({start}) must be less than 'end' ({end_})")
    if end_ > len(transforms):
        raise ValueError(f"'end' ({end_}) must be less than or equal to the transform count ({len(transforms)}")

    # no-op if the range is empty
    if start == end:
        return data

    for _transform in transforms[start:end]:
        if threading:
            _transform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
        data = apply_transform(
            _transform, data, map_items, unpack_items, lazy=lazy, overrides=overrides, log_stats=log_stats
        )
    data = apply_pending_transforms(data, None, overrides, logger_name=log_stats)
    return data

class Compose(Randomizable, InvertibleTransform, LazyTransform):
    """
    `Compose` provides the ability to chain a series of callables together in
    a sequential manner. Each transform in the sequence must take a single
    argument and return a single value.

    ... (your docstring remains unchanged) ...
    """

    def __init__(
        self,
        transforms: [Sequence[Callable], Callable, None] = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: [bool, str] = False,
        lazy: [bool, None] = False,
        overrides: [dict, None] = None,
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)

        if transforms is None:
            transforms = []

        if not isinstance(map_items, bool):
            raise ValueError(
                f"Argument 'map_items' should be boolean. Got {type(map_items)}."
                "Check brackets when passing a sequence of callables."
            )

        self.transforms = ensure_tuple(transforms)
        self.map_items = map_items
        self.unpack_items = unpack_items
        self.log_stats = log_stats
        self.set_random_state()
        self.overrides = overrides

    @LazyTransform.lazy.setter
    def lazy(self, val: bool):
        self._lazy = val

    def set_random_state(self, seed: [int, None] = None, state: [np.random.RandomState, None] = None) -> 'Compose':
        super().set_random_state(seed=seed, state=state)
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            _transform.set_random_state()
        return self

    def randomize(self, data: [Any, None] = None) -> None:
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            try:
                _transform.randomize(data)
            except TypeError as type_error:
                tfm_name: str = type(_transform).__name__
                warnings.warn(
                    f"Transform '{tfm_name}' in Compose not randomized\n{tfm_name}.{type_error}.", RuntimeWarning
                )

    def get_index_of_first(self, predicate):
        for i in range(len(self.transforms)):
            if predicate(self.transforms[i]):
                return i
        return None

    def flatten(self):
        new_transforms = []
        for t in self.transforms:
            if isinstance(t, Compose):
                new_transforms += t.flatten().transforms
            else:
                new_transforms.append(t)

        return Compose(new_transforms)

    def __len__(self):
        return len(self.flatten().transforms)

    def __call__(self, input, start=0, end=None, threading=False, lazy: [bool, None] = None):
        _lazy = self._lazy if lazy is None else lazy
        result = execute_compose(
            input,
            transforms=self.transforms,
            start=start,
            end=end,
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            lazy=_lazy,
            overrides=self.overrides,
            threading=threading,
            log_stats=self.log_stats,
        )

        return result

    def inverse(self, data):
        self._raise_if_not_invertible(data)

        invertible_transforms = [t for t in self.flatten().transforms if isinstance(t, InvertibleTransform)]
        if not invertible_transforms:
            warnings.warn("inverse has been called but no invertible transforms have been supplied")

        if self._lazy is True:
            warnings.warn(
                f"'lazy' is set to {self._lazy} but lazy execution is not supported when inverting. "
                f"'lazy' has been overridden to False for the call to inverse"
            )
        for t in reversed(invertible_transforms):
            data = apply_transform(
                t.inverse, data, self.map_items, self.unpack_items, lazy=False, log_stats=self.log_stats
            )
        return data

    @staticmethod
    def _raise_if_not_invertible(data: Any):
        from monai.transforms.utils import has_status_keys

        invertible, reasons = has_status_keys(
            data, TraceStatusKeys.PENDING_DURING_APPLY, "Pending operations while applying an operation"
        )

        if not invertible:
            if reasons is not None:
                reason_text = "\n".join(reasons)
                raise RuntimeError(f"Unable to run inverse on 'data' for the following reasons:\n{reason_text}")
            else:
                raise RuntimeError("Unable to run inverse on 'data'; no reason logged in trace data")


source_transforms = Compose(
    [
        LoadImaged(keys=["img", "brain_mask"]),
        EnsureChannelFirstd(keys=["img",  "brain_mask"]),
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),
        RandSpatialCropd(keys=["img","brain_mask"], roi_size=(112, 112, 112), random_size=False),
        #RandCropByPosNegLabeld(
        #    keys=["img", "brain_mask"],
        #    spatial_size=(64, 64, 64),
        #    label_key="brain_mask",
        #    pos = 0.9,
        #    neg=0.1,
        #    num_samples=1,
        #    image_key="img",
        #    image_threshold=-0.1
        #),
        #AdjustContrastd(keys=["img"], gamma=2.0),
        RandAxisFlipd(keys=["img", "brain_mask"], prob = 0.2),
        RandGaussianNoised(keys = ["img"], prob=0.2, mean=0.0, std=0.05),
        RandGibbsNoised(keys=["img"], prob = 0.2, alpha = (0.1,0.6))
    ]
)

def threshold(x):
    # threshold at 1
    return x > 0.015


target_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
        CropForegroundd(keys=["img"], source_key = "img", select_fn=threshold, margin=3),
        RandSpatialCropd(keys=["img"], roi_size=(112, 112, 112), random_size=False),
        RandGaussianNoised(keys = ["img"], prob=0.2, mean=0.0, std=0.05),
        RandGibbsNoised(keys=["img"], prob = 0.2, alpha = (0.1,0.6)),
        RandAxisFlipd(keys=["img"], prob = 0.2)
    ]
)


def load_data(source_dev_images_csv, source_dev_masks_csv,
              target_dev_images_csv = None, batch_size = 1, val_split = 0.2, verbose = False):


    source_dev_images = pd.read_csv(source_dev_images_csv)
    source_dev_masks = pd.read_csv(source_dev_masks_csv)

    assert source_dev_images.size == source_dev_masks.size

    if target_dev_images_csv:
        target_dev_images = pd.read_csv(target_dev_images_csv)

    if verbose:
        print("Shape source images:", source_dev_images.shape)
        print("Shape source masks:",  source_dev_masks.shape)
        if target_dev_images_csv:
            print("Shape target images:", target_dev_images.shape)
        else:
            print("Target images CSV file path not provided")    
    
    
    indexes_source = np.arange(source_dev_images.shape[0])
    
    np.random.seed(100)  
    np.random.shuffle(indexes_source)
    
  
    source_dev_images = np.array(source_dev_images["filename"])[indexes_source]
    source_dev_masks = np.array(source_dev_masks["filename"])[indexes_source]
    
    ntrain_samples = int((1 - val_split)*indexes_source.size)
    source_train_images = source_dev_images[:ntrain_samples]
    source_train_masks = source_dev_masks[:ntrain_samples]

    source_val_images = source_dev_images[ntrain_samples:]
    source_val_masks = source_dev_masks[ntrain_samples:]

    if verbose:
        print("Source train set size:", source_train_images.size)
        print("Source val set size:", source_val_images.size)


    # Putting the filenames in the MONAI expected format - source train set
    filenames_train_source = [{"img": x, "brain_mask": y, "domain_label": 0.0}\
                              for (x,y) in zip(source_train_images, source_train_masks)]
       
    source_ds_train = monai.data.Dataset(filenames_train_source,
                                         source_transforms)

    source_train_loader = DataLoader(source_ds_train, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here


    # Putting the filenames in the MONAI expected format - source val set
    filenames_val_source = [{"img": x, "brain_mask": y, "domain_label": 0.0}\
                              for (x,y) in zip(source_val_images, source_val_masks)]
       
    source_ds_val = monai.data.Dataset(filenames_val_source,
                                         source_transforms)
                                         
    source_val_loader = DataLoader(source_ds_val, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here



    # If there is not target domain data - return the source domain train and val datasets and loaders
    if not target_dev_images_csv:
        return source_ds_train, source_train_loader, source_ds_val, source_val_loader

    
    
    indexes_target = np.arange(target_dev_images.shape[0])
    np.random.seed(100)  
    np.random.shuffle(indexes_target)

    target_dev_images = np.array(target_dev_images["filename"])[indexes_target]
    
    ntrain_samples_target = int((1 - val_split)*indexes_target.size)
    target_train_images = target_dev_images[:ntrain_samples_target]
    
    target_val_images = target_dev_images[ntrain_samples_target:]

    if verbose:
        print("Traget train set size:", target_train_images.size)
        print("Target val set size:", target_val_images.size)


    # Putting the filenames in the MONAI expected format - target train set
    filenames_train_target = [{"img": x, "domain_label": 1.0}\
                              for x in target_train_images]
       
    target_ds_train = monai.data.Dataset(filenames_train_target,
                                         target_transforms)

    target_train_loader = DataLoader(target_ds_train, 
                                    batch_size=batch_size, 
                                    shuffle = True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here

    # Putting the filenames in the MONAI expected format - target val set
    filenames_val_target = [{"img": x, "domain_label": 1.0}\
                              for x in target_val_images]


    target_ds_val = monai.data.Dataset(filenames_val_target,
                                         target_transforms)
                                         
    target_val_loader = DataLoader(target_ds_val, 
                                   batch_size=batch_size, 
                                   shuffle = True, 
                                   num_workers=0, 
                                   pin_memory=True, 
                                   collate_fn=pad_list_data_collate,
                                   drop_last=True) # add drop_last argument here

    return source_ds_train, source_train_loader, source_ds_val, source_val_loader,\
           target_ds_train, target_train_loader, target_ds_val, target_val_loader

