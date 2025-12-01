import typing as t
import os
import h5py
from functools import partial

from pydantic.dataclasses import dataclass

from pyfm import utils


@dataclass(frozen=True)
class LoadArrayConfig:
    """Parameters providing index names and values to convert ndarray
    into a DataFrame.

    Properties
    ----------
        order: list
            A list of names given to each dimension of a
            multidimensional ndarray in the order of the array's shape
        labels: dict(str, Union(str, list))
            labels for each index of multidimensional array. Dictionary
            keys should match entries in `order`. Dictionary values
            should either be lists with length matching corresponding
            dimension of ndarray, or a string range in the format 'start..stop'
            where start-stop = length of array for the given dimension,
            (note: stop is inclusive).
    """

    order: t.List
    labels: t.Dict[str, t.Union[str, t.List]]

    @classmethod
    def create(
        cls,
        order: t.List | None = None,
        labels: t.Dict | None = None,
        array_order: t.List | None = None,
        array_labels: t.Dict | None = None,
    ) -> "LoadArrayConfig":
        o = array_order or order
        l = array_labels or labels
        assert o and l, "Must provide an 'order' and 'labels' parameter"
        return cls(
            order=o,
            labels=utils.string.process_params(**l),
        )


@dataclass(frozen=True)
class LoadDictConfig:
    labels: t.List[str]
    array_config: LoadArrayConfig

    @classmethod
    def create(
        cls,
        labels: t.List[str] | None = None,
        dict_labels: t.List[str] | None = None,
        *args,
        **kwargs,
    ) -> "LoadDictConfig":
        l = dict_labels or labels
        assert l, "Must provide 'labels' parameter"
        return cls(labels=l, array_config=LoadArrayConfig.create(*args, **kwargs))


@dataclass(frozen=True)
class LoadH5Config:
    name: str
    datasets: t.Dict[str, t.List[str]]
    array_config: t.Dict[str, LoadArrayConfig]

    @classmethod
    def create(cls, name: str, datasets: t.Dict, *args, **kwargs) -> "LoadH5Config":
        dsets = {}
        for k, v in datasets.items():
            if isinstance(datasets[k], str):
                dsets[k] = [v]
            else:
                dsets[k] = v

        array_config = {}
        # TODO: Currently only supports having same array params (encoded in args, kwargs)
        # for all h5 data sets
        for k in datasets.keys():
            array_config[k] = LoadArrayConfig.create(*args, **kwargs)

        return cls(name=name, datasets=dsets, array_config=array_config)

    def search_for_dataset_label(self, file: h5py.File):
        ds = {k: [] for k in self.datasets.keys()}

        def visitor(name, obj, target, agg):
            if not isinstance(target, t.List) or len(target) != 1:
                raise ValueError("Target value is not a list[str] with a single entry.")

            if isinstance(obj, h5py.Dataset):
                # Check if the dataset's base name matches target
                if name.split("/")[-1] == target[0]:
                    agg.append(name)

        for k, v in ds.items():
            file.visititems(partial(visitor, target=self.datasets[k], agg=v))
        h5_params = {
            "name": self.name,
            "datasets": ds,
            "array_config": self.array_config.copy(),
        }

        return type(self)(**h5_params)

    def format_data_strings(self, repl: t.Dict[str, str]):
        h5_params = {
            "name": self.name,
            "datasets": {
                k.format(**repl): [vv.format(**repl) for vv in v]
                for k, v in self.datasets.items()
            },
            "array_config": {k.format(**repl): v for k, v in self.array_config.items()},
        }

        return type(self)(**h5_params)
