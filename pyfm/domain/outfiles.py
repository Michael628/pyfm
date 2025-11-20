from dataclasses import dataclass
import typing as t
import os


@dataclass
class Outfile:
    filestem: str
    ext: str
    good_size: int

    @staticmethod
    def from_param(
        file_label: str, file_path: str, file_config: t.Dict[str, t.Any]
    ) -> "Outfile":
        def get_extension() -> str:
            extensions = {
                "cfg": ".{cfg}",
                "cfg_bin": ".{cfg}.bin",
                "cfg_bin_multi": ".{cfg}/v{eig_index}.bin",
                "cfg_h5": ".{cfg}.h5",
                "cfg_gamma_h5": ".{cfg}/{gamma}_0_0_0.h5",
                "cfg_pickle": ".{cfg}.p",
            }

            name_keys = {
                "links": "cfg",
                "meson": "cfg_gamma_h5",
                "eig": "cfg_bin",
                "a2a_vec": "cfg_bin",
                "contract_legacy": "cfg_pickle",
                "contract": "cfg_h5",
                "modes": "cfg_h5",
                "eval": "cfg_h5",
                "edir": "cfg_bin_multi",
            }
            key = None
            for k, v in name_keys.items():
                if k in file_label:
                    key = v
                    break
            if key is None:
                raise ValueError(f"No outfile definition for {file_label}.")

            return extensions[key]

        if filestem := file_config.get("filestem", None):
            filestem = str(os.path.join(file_path, filestem))
        else:
            raise ValueError(f"No filestem provided for {file_label}")

        if good_size := file_config.get("good_size", None):
            good_size = good_size
        else:
            raise ValueError(f"No good_size provided for {file_label}")

        ext = get_extension()

        return Outfile(filestem, ext, good_size)

    @property
    def filename(self) -> str:
        return self.filestem + self.ext

    def format_map(self, formatter: t.Dict[str, str]) -> "Outfile":
        return Outfile(
            filestem=self.filestem.format_map(formatter),
            ext=self.ext.format_map(formatter),
            good_size=self.good_size,
        )
