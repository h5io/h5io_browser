from concurrent.futures import Future

from h5io_browser.base import list_hdf, read_dict_from_hdf


class HDFFuture(Future):
    def __init__(self, file_name, h5_path):
        super().__init__()
        self._file_name = file_name
        self._h5_path = h5_path 

    def done(self):
        return True

    def result(self):
        return read_dict_from_hdf(
            file_name=self._file_name, 
            h5_path=self._h5_path,
        )[self._h5_path.split("/")[-1]]
    

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
    return {
        k.replace(h5_path + "/", ""): HDFFuture(file_name=file_name, h5_path=k)
        for k in list_hdf(
            file_name=file_name, h5_path=h5_path, recursive=recursive
        )[0]
    }
