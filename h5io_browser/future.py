import os
import warnings
from concurrent.futures import Future
from typing import Optional

from h5io_browser.base import list_hdf, read_dict_from_hdf


class HDFFuture(Future):
    def __init__(
        self, file_name: str, h5_path: str, file_time_stamp: Optional[float] = None
    ):
        super().__init__()
        self._file_name = file_name
        self._h5_path = h5_path
        self._hdf_read = False
        if file_time_stamp is None:
            self._file_time_stamp = os.stat(file_name).st_mtime
        else:
            self._file_time_stamp = file_time_stamp

    def _load_from_hdf(self):
        if not self._hdf_read:
            if os.stat(self._file_name).st_mtime != self._file_time_stamp:
                warnings.warn(
                    "The HDF5 file has been modified since the Future object was initialized: {}".format(
                        self._file_name
                    )
                )
            try:
                self.set_result(
                    read_dict_from_hdf(
                        file_name=self._file_name,
                        h5_path=self._h5_path,
                    )[self._h5_path.split("/")[-1]]
                )
            except Exception as file_access_exception:
                self.set_exception(file_access_exception)
            self._hdf_read = True

    def done(self):
        self._load_from_hdf()
        return super().done()

    def result(self, timeout=None):
        self._load_from_hdf()
        return super().result(timeout=timeout)


def read_future_dict_from_hdf(
    file_name: str,
    h5_path: str,
    recursive: bool = False,
):
    """
    Read data from HDF5 file into a dictionary - by default only the nodes are converted to dictionaries, additional
    sub groups can be specified either using the group_paths parameter or using the recursive parameter.

    Args:
        file_name (str): Name of the file on disk
        h5_path (str): Path to a group in the HDF5 file from where the data is read
        recursive (bool/int): Recursively browse through the HDF5 file, either a boolean flag or an integer
                              which specifies the level of recursion.

    Returns:
        dict: The loaded data as nested dictionary with the results being represented as Future objects.
    """
    file_time_stamp = os.stat(file_name).st_mtime
    return {
        k.replace(h5_path + "/", ""): HDFFuture(
            file_name=file_name, h5_path=k, file_time_stamp=file_time_stamp
        )
        for k in list_hdf(file_name=file_name, h5_path=h5_path, recursive=recursive)[0]
    }
