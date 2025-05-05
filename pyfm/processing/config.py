import typing as t

from pydantic.dataclasses import dataclass

from pyfm import ConfigBase, utils


@dataclass
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

    def __post_init__(self):
        self.labels = utils.process_params(**self.labels)


@dataclass
class LoadDictConfig:
    labels: t.List[str]
    array_config: LoadArrayConfig

    def __init__(
        self,
        labels: t.List[str],
        array_params: t.Dict,
    ) -> None:
        self.labels = labels
        self.array_config = LoadArrayConfig(**array_params)


@dataclass
class LoadH5Config:
    name: str
    datasets: t.Dict[str, t.List[str]]
    array_config: t.Dict[str, LoadArrayConfig]

    @classmethod
    def create(
        cls,
        name: str,
        datasets: t.Dict,
        array_params: t.Dict,
    ) -> "LoadH5Config":
        assert all((k in array_params for k in datasets.keys())), (
            "Must define array_params for each data set"
        )

        dsets = {}
        for k, v in datasets.items():
            if isinstance(datasets[k], str):
                dsets[k] = [v]
            else:
                dsets[k] = v

        array_config = {}
        for k, v in array_params.items():
            array_config[k] = LoadArrayConfig(**v)

        return cls(name=name, datasets=dsets, array_config=array_config)

    @classmethod
    def string_replace(cls, h5_config: "LoadH5Config", repl: t.Dict[str, str]):
        h5_params = {
            "name": h5_config.name,
            "datasets": {
                k.format(**repl): [vv.format(**repl) for vv in v]
                for k, v in h5_config.datasets.items()
            },
            "array_config": {
                k.format(**repl): v for k, v in h5_config.array_config.items()
            },
        }

        return cls(**h5_params)


@dataclass
class DataioConfig(ConfigBase):
    filestem: str
    replacements: t.Optional[t.Dict[str, t.Union[str, t.List[str]]]] = None
    regex: t.Optional[t.Dict[str, str]] = None
    dict_labels: t.Optional[t.List[str]] = None
    actions: t.Optional[t.Dict[str, t.Any]] = None

    def __post_init__(self):
        """Set defaults and process `replacements`"""

        if not self.replacements:
            self.replacements = {}
        if not self.regex:
            self.regex = {}
        if not self.dict_labels:
            self.dict_labels = []
        if not self.actions:
            self.actions = {}

        self.replacements = utils.process_params(**self.replacements)

    @classmethod
    def create(cls, **kwargs):
        """Returns an instance of DataioConfig from `params` dictionary.

        Parameters
        ----------
        kwargs
            keys should correspond to class parameters (above).
            `h5_params` and `array_params`, if provided,
            should have dictionaries that can be passed to `create` static methods
            in H5Params and ArrayParams, respectively.
        """

        obj_vars = kwargs.copy()
        array_params = obj_vars.pop("array_params", {})

        return DataioConfig(**obj_vars)


def get_config_factory(config_label: str):
    configs = {"dataio": DataioConfig.create}

    if config_label in configs:
        return configs[config_label]
    else:
        raise ValueError(f"No config implementation for `{config_label}`.")


def get_dataio_config(params: t.Dict) -> DataioConfig:
    return get_config_factory("dataio")(**params)
