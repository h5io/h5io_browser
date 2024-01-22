import os
import numpy as np
import h5io
import h5py
from itertools import count
import sys
import time
from typing import Callable, TypeVar, Type, Tuple, Optional, Union
import warnings


T = TypeVar("T")


def delete_item(file_name, h5_path):
    """
    Delete item from HDF5 file

    Args:
       file_name (str): Name of the file on disk
       h5_path (str): Path to a group in the HDF5 file from where the data is read
    """
    try:
        if os.path.exists(file_name):
            with _open_hdf(file_name, mode="a") as store:
                del store[h5_path]
    except (AttributeError, KeyError):
        pass


def list_hdf(file_name, h5_path, recursive=False):
    """
    List HDF5 nodes and HDF5 groups of a given HDF5 file at a given h5_path

    Args:
       file_name (str): Name of the file on disk
       h5_path (str): Path to a group in the HDF5 file from where the data is read
       recursive (bool/int): Recursively browse through the HDF5 file, either a boolean flag or an integer
                              which specifies the level of recursion.

    Returns:
       (list, list): list of HDF5 nodes and list of HDF5 groups
    """
    if os.path.exists(file_name):
        with h5py.File(file_name, "r") as hdf:
            try:
                return _get_hdf_content(hdf=hdf[h5_path], recursive=recursive)
            except KeyError:
                return [], []
    else:
        return [], []


def read_dict_from_hdf(file_name, h5_path, recursive=False, slash="ignore"):
    """
    Read data from HDF5 file into a dictionary - by default only the nodes are converted to dictionaries, additional
    sub groups can be specified using the group_paths parameter.

    Args:
       file_name (str): Name of the file on disk
       h5_path (str): Path to a group in the HDF5 file from where the data is read
       recursive (bool/int): Recursively browse through the HDF5 file, either a boolean flag or an integer
                              which specifies the level of recursion.
       slash (str): 'ignore' | 'replace' Whether to replace the string {FWDSLASH} with the value /. This does
                    not apply to the top level name (title). If 'ignore', nothing will be replaced.
    Returns:
       dict:     The loaded data. Can be of any type supported by ``write_hdf5``.
    """
    with h5py.File(file_name, "r") as hdf:
        if recursive:
            nodes_lst = _get_hdf_content(
                hdf=hdf[h5_path], recursive=recursive, only_nodes=True
            )
        else:
            nodes_lst = [h5_path]
        if len(nodes_lst) > 0 and nodes_lst[0] != "/":
            return {
                n: _read_hdf(hdf_filehandle=hdf, h5_path=n, slash=slash)
                for n in nodes_lst
            }
        else:
            return {}


def write_dict_to_hdf(file_name, data_dict, compression=4, slash="error"):
    """
    Write dictionary to HDF5 file

    Args:
        file_name (str): Name of the file on disk
        data_dict (dict): Dictionary of data objects to be stored in the HDF5 file, the keys provide the path inside
                          the HDF5 file and the values the data to be stored in those nodes. The corresponding HDF5
                          groups are created automatically:
                              {
                                  '/hdf5root/group/node_name': {},
                                  '/hdf5root/group/subgroup/node_name': [...],
                              }
        compression (int): Compression level to use (0-9) to compress data using gzip.
        slash (str): 'error' | 'replace' Whether to replace forward-slashes ('/') in any key found nested within
                      keys in data. This does not apply to the top level name (title). If 'error', '/' is not allowed
                      in any lower-level keys.
    """
    with _open_hdf(file_name, mode="a") as store:
        for k, v in data_dict.items():
            _write_hdf5_with_json_support(
                hdf_filehandle=store,
                data=v,
                h5_path=k,
                compression=compression,
                slash=slash,
            )


def _get_filename_from_filehandle(hdf_filehandle):
    """
    Get filename from filehandle object

    Args:
        hdf_filehandle (h5py.File): Open h5py file handle

    Returns:
        str: filename of the filehandle object
    """
    if isinstance(hdf_filehandle, h5py.File):
        file_name = hdf_filehandle.filename
    elif isinstance(hdf_filehandle, str):
        file_name = hdf_filehandle
    else:
        raise TypeError(
            "The hdf_filehandle should either be the file name as string or the h5py.File object: ",
            type(hdf_filehandle),
        )
    return file_name


def _open_hdf(filename, mode="r", swmr=False):
    """
    Open HDF5 file

    Args:
        filename (str): Name of the file on disk, or file-like object.  Note: for files created with the 'core' driver,
                        HDF5 still requires this be non-empty.
        mode (str): r        Readonly, file must exist (default)
                    r+       Read/write, file must exist
                    w        Create file, truncate if exists
                    w- or x  Create file, fail if exists
                    a        Read/write if exists, create otherwise
        swmr (bool): Open the file in SWMR read mode. Only used when mode = 'r'.

    Returns:
        h5py.File: open HDF5 file object
    """
    if swmr and mode != "r":
        store = h5py.File(name=filename, mode=mode, libver="latest")
        store.swmr_mode = True
        return store
    else:
        return h5py.File(name=filename, mode=mode, libver="latest", swmr=swmr)


def _read_hdf(hdf_filehandle, h5_path, slash="ignore"):
    """
    Read data from HDF5 file

    Args:
        hdf_filehandle (str/h5py.File): Open h5py file handle or file name as string
        h5_path (str): the HDF5 internal dataset path from which should be read, slashes indicate sub groups
        slash (str): 'ignore' | 'replace' Whether to replace the string {FWDSLASH} with the value /. This does
                     not apply to the top level name (title). If 'ignore', nothing will be replaced.

    Returns:
        object:     The loaded data. Can be of any type supported by ``write_hdf5``.
    """
    file_name = _get_filename_from_filehandle(hdf_filehandle=hdf_filehandle)
    return _retry(
        lambda: h5io.read_hdf5(
            fname=hdf_filehandle,
            title=h5_path,
            slash=slash,
        ),
        error=BlockingIOError,
        msg=f"Two or more processes tried to access the file {file_name}.",
        at_most=10,
        delay=1,
    )


def _write_hdf(
    hdf_filehandle,
    h5_path,
    data,
    overwrite=False,
    compression=4,
    slash="error",
    use_json=False,
    use_state=False,
):
    """
    Write data to HDF5 file

    Args:
        hdf_filehandle (str/h5py.File): Open h5py file handle or file name as string
        h5_path (str): the HDF5 internal dataset path from which should be read, slashes indicate sub groups
        data (object): Object to write. Can be of any of these types: {ndarray, dict, list, tuple, int, float, str,
                       datetime, timezone} Note that dict objects must only have ``str`` keys. It is recommended
                       to use ndarrays where possible, as it is handled most efficiently.
        overwrite (str/bool): True | False | 'update' If True, overwrite file (if it exists). If 'update', appends the
                              title to the file (or replace value if title exists).
        compression (int): Compression level to use (0-9) to compress data using gzip.
        slash (str): 'error' | 'replace' Whether to replace forward-slashes ('/') in any key found nested within
                      keys in data. This does not apply to the top level name (title). If 'error', '/' is not allowed
                      in any lower-level keys.
        use_json (bool): To accelerate the read and write performance of small dictionaries and lists they can be
                         combined to JSON objects and stored as strings.
        use_state (bool): To store objects of unsupported types the __getstate__() method is used to retrieve a
                          dictionary which defines the state of the object and store the content of this dictionary in
                          the HDF5 file. (requires python >=3.11)
    """
    file_name = _get_filename_from_filehandle(hdf_filehandle=hdf_filehandle)
    _retry(
        lambda: h5io.write_hdf5(
            fname=hdf_filehandle,
            data=data,
            overwrite=overwrite,
            compression=compression,
            title=h5_path,
            slash=slash,
            use_json=use_json,
            use_state=use_state,
        ),
        error=BlockingIOError,
        msg=f"Two or more processes tried to access the file {file_name}.",
        at_most=10,
        delay=1,
    )


def _write_hdf5_with_json_support(
    hdf_filehandle, h5_path, data, compression=4, slash="error"
):
    """
    Write data to HDF5 file and store dictionaries as JSON to optimize performance

    Args:
        hdf_filehandle (str/h5py.File): Open h5py file handle or file name as string
        h5_path (str): the HDF5 internal dataset path from which should be read, slashes indicate sub groups
        data (object): Object to write. Can be of any of these types: {ndarray, dict, list, tuple, int, float, str,
                       datetime, timezone} Note that dict objects must only have ``str`` keys. It is recommended
                       to use ndarrays where possible, as it is handled most efficiently.
        compression (int): Compression level to use (0-9) to compress data using gzip.
        slash (str): 'error' | 'replace' Whether to replace forward-slashes ('/') in any key found nested within
                      keys in data. This does not apply to the top level name (title). If 'error', '/' is not allowed
                      in any lower-level keys.
    """
    value, use_json = _check_json_conversion(value=data)
    try:
        _write_hdf(
            hdf_filehandle=hdf_filehandle,
            h5_path=h5_path,
            data=value,
            compression=compression,
            slash=slash,
            use_json=use_json,
            use_state=sys.version_info >= (3, 11),
            overwrite="update",
        )
    except TypeError:
        raise TypeError(
            "Error saving {} (key {}): h5io_browser doesn't support saving elements "
            'of type "{}" to HDF!'.format(value, h5_path, type(value))
        ) from None


def _list_h5path(hdf):
    """
    List groups and nodes in a given HDF5 path

    Args:
        hdf (h5py.File/h5py.Group): HDF5 pointer

    Returns:
        (list, list): list of groups and list of nodes
    """
    group_lst, nodes_lst = [], []
    try:
        for k in hdf.keys():
            if isinstance(hdf[k], h5py.Group):
                group_lst.append(hdf[k].name)
            else:
                nodes_lst.append(hdf[k].name)
    except (AttributeError, KeyError):
        return [], []
    else:
        return nodes_lst, group_lst


def _get_hdf_content(hdf, recursive=False, only_groups=False, only_nodes=False):
    """
    Get all sub-groups of a given HDF5 path

    Args:
        hdf (h5py.File/h5py.Group): HDF5 pointer
        recursive (bool/int): Recursively browse through the HDF5 file, either a boolean flag or an integer
                              which specifies the level of recursion.
        only_groups (bool): return only HDF5 groups
        only_nodes (bool): return only HDF5 nodes

    Returns:
        list/(list, list): list of HDF5 groups or list of HDF5 nodes or tuple of both lists
    """
    if isinstance(recursive, bool):
        recursive_flag = recursive
    elif isinstance(recursive, int) and recursive > 0:
        recursive_flag = True
    elif isinstance(recursive, int) and recursive <= 0:
        recursive_flag = False
    else:
        raise TypeError(
            "The recursive parameter, has to be either bool or int: ",
            recursive,
            type(recursive),
        )

    if recursive_flag:
        if not isinstance(recursive, bool) and isinstance(recursive, int):
            recursive -= 1
        group_lst = []
        nodes_lst, groups_to_iterate_lst = _list_h5path(hdf=hdf)
        for group in groups_to_iterate_lst:
            nodes, groups = _get_hdf_content(hdf=hdf[group], recursive=recursive)
            nodes_lst += nodes
            group_lst += [group] + groups
        if only_groups:
            return group_lst
        elif only_nodes:
            return nodes_lst
        else:
            return nodes_lst, group_lst
    elif only_groups:
        return _list_h5path(hdf=hdf)[1]
    elif only_nodes:
        return _list_h5path(hdf=hdf)[0]
    else:
        return _list_h5path(hdf=hdf)


def _check_json_conversion(value):
    """
    Check if the object can be converted to JSON to optimize the HDF5 performance. This can change the data type of the
    object which is going to be stored in the HDF5 file.

    Args:
        value (object): Object to be converted.

    Returns:
        object bool: value object and boolean flag to store the dictionary as JSON
    """
    use_json = True
    if (
        isinstance(value, (list, np.ndarray))
        and len(value) > 0
        and isinstance(value[0], (list, np.ndarray))
        and len(value[0]) > 0
        and not isinstance(value[0][0], str)
        and _is_ragged_in_1st_dim_only(value)
    ):
        # if the sub-arrays in value all share shape[1:], h5io comes up with a more efficient storage format than
        # just writing a dataset for each element, by concatenating along the first axis and storing the indices
        # where to break the concatenated array again
        value = np.array([np.asarray(v) for v in value], dtype=object)
        use_json = False
    elif isinstance(value, tuple):
        value = list(value)
    return value, use_json


def _is_ragged_in_1st_dim_only(value: Union[np.ndarray, list]) -> bool:
    """
    Checks whether array or list of lists is ragged in the first dimension.

    That means all other dimensions (except the first one) still have to match.

    Args:
        value (ndarray/list): array to check

    Returns:
        bool: True if elements of value are not all of the same shape
    """
    if isinstance(value, np.ndarray) and value.dtype != np.dtype("O"):
        return False
    else:

        def extract_dims(v):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = np.shape(v)
            return s[0], s[1:]

        dim1, dim_other = zip(*map(extract_dims, value))
        return len(set(dim1)) > 1 and len(set(dim_other)) == 1


def _retry(
    func: Callable[[], T],
    error: Union[Type[Exception], Tuple[Type[Exception], ...]],
    msg: str,
    at_most: Optional[int] = None,
    delay: float = 1.0,
    delay_factor: float = 1.0,
) -> T:
    """
    Try to call `func` until it no longer raises `error`.

    Any other exception besides `error` is still raised.

    Args:
        func (callable): function to call, should take no arguments
        error (Exception or tuple thereof): any exceptions to be caught
        msg (str): messing to be written to the log if `error` occurs.
        at_most (int, optional): retry at most this many times, None means retry
                                forever
        delay (float): time to wait between retries in seconds
        delay_factor (float): multiply `delay` between retries by this factor

    Raises:
        `error`: if `at_most` is exceeded the last error is re-raised
        Exception: any exception raised by `func` that does not match `error`

    Returns:
        object: whatever is returned by `func`
    """
    if at_most is None:
        tries = count()
    else:
        tries = range(at_most)
    for i in tries:
        try:
            return func()
        except error as e:
            warnings.warn(
                f"{msg} Trying again in {delay}s. Tried {i + 1} times so far..."
            )
            time.sleep(delay)
            delay *= delay_factor
            # e drops out of the namespace after the except clause ends, so
            # assign it here to a dummy variable so that we can re-raise it
            # in case the error persists
            err = e
    raise err from None
