import os
import posixpath
from collections.abc import MutableMapping
from typing import Any, Dict, List, Union

import h5py

from h5io_browser.base import (
    _open_hdf,
    _read_dict_from_open_hdf,
    delete_item,
    list_hdf,
    write_dict_to_hdf,
)


def convert_dict_items_to_str(input_dict: dict) -> dict:
    """
    Recursively convert all keys and values to strings using the str() method.

    Args:
        input_dict (dict): dictionary to be converted, can be hierarchical

    Returns:
        dict: dictionary with all keys being strings and all values being either dictionaries or also strings on all
              levels of the hierarchy.
    """
    return {
        str(k): (
            convert_dict_items_to_str(input_dict=v) if isinstance(v, dict) else str(v)
        )
        for k, v in input_dict.items()
    }


def get_hierarchical_dict(path_dict: dict) -> dict:
    """
    Convert a flat dictionary which consists of the keys as HDF5 paths and the values as the data stored in these HDF5
    nodes to a hierarchical dictionary.

    Examples:

        >>> get_hierarchical_dict(path_dict={
        >>>     "test/path/a": 1,
        >>>     "test/path/b": [2, 3],
        >>>     "test/path/c/d": 4,
        >>>     "test/path/c/e": {"f": 5},
        >>> })
        {"test": {"path": {"a": 1, "b": [2, 3], "c": {"d": 4, "e": {"f": 5}}}}}

    Args:
        path_dict (dict): flat dictionary with h5_paths as keys

    Returns:
        dict: hierarchical dictionary
    """
    summary_dict = {}
    for k, v in path_dict.items():
        if "/" in k:
            k_split_lst = k.split("/")
            if k_split_lst[0] == "":
                k_split_lst = k_split_lst[1:]
            if len(k_split_lst) > 1:
                current_dict = summary_dict
                for ks in k_split_lst[:-1]:
                    if ks not in current_dict.keys():
                        current_dict[ks] = {}
                    current_dict = current_dict[ks]
                current_dict[k_split_lst[-1]] = v
            else:
                summary_dict[k] = v
        else:
            summary_dict[k] = v
    return summary_dict


class Pointer(MutableMapping):
    def __init__(self, file_name: str, h5_path: str = "/") -> None:
        """
        Initialize the Pointer object.

        Args:
            file_name (str): The name of the HDF5 file.
            h5_path (str, optional): The path in the HDF5 file starting from the root group. Defaults to "/".
        """
        file_name += ".h5" if not file_name.endswith(".h5") else ""
        self._file_name = None
        self._h5_path = None
        self.file_name = file_name
        self.h5_path = h5_path

    @property
    def file_exists(self) -> bool:
        """
        Check if the HDF5 file exists already.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        if os.path.isfile(self.file_name):
            return True
        else:
            return False

    @property
    def file_name(self) -> str:
        """
        Get the file name of the HDF5 file.

        Returns:
            str: The absolute path to the HDF5 file.
        """
        return self._file_name

    @file_name.setter
    def file_name(self, new_file_name: str) -> None:
        """
        Set the file name of the HDF5 file.

        Args:
            new_file_name (str): The absolute path to the HDF5 file.
        """
        self._file_name = os.path.abspath(new_file_name).replace("\\", "/")

    @property
    def h5_path(self) -> str:
        """
        Get the path in the HDF5 file starting from the root group.

        Returns:
            str: The HDF5 path.
        """
        return self._h5_path

    @h5_path.setter
    def h5_path(self, path: str) -> None:
        """
        Set the path in the HDF5 file starting from the root group - meaning this path starts with '/'.

        Args:
            path (str): The HDF5 path.
        """
        if (path is None) or (path == ""):
            path = "/"
        self._h5_path = posixpath.normpath(path)
        if not posixpath.isabs(self._h5_path):
            self._h5_path = "/" + self._h5_path

    @property
    def is_empty(self) -> bool:
        """
        Check if the HDF5 file is empty.

        Returns:
            bool: True if the file is empty, False otherwise.
        """
        if self.file_exists:
            nodes, groups = list_hdf(
                file_name=self.file_name, h5_path="/", recursive=True
            )
            return (len(nodes) + len(groups)) == 0
        else:
            return True

    @property
    def is_root(self) -> bool:
        """
        Check if the current h5_path is pointing to the HDF5 root group.

        Returns:
            bool: True if the current h5_path is the root group, False otherwise.
        """
        return "/" == self.h5_path

    def copy_to(
        self,
        destination: "Pointer",
        file_name: str = None,
        maintain_name: bool = True,
    ) -> "Pointer":
        """
        Copy the content of the HDF5 file to a new location.

        Args:
            destination (Pointer): The Pointer object pointing to the new location.
            file_name (str, optional): The name of the new HDF5 file. Defaults to None.
            maintain_name (bool, optional): Whether to maintain the names of the HDF5 groups. Defaults to True.

        Returns:
            Pointer: The Pointer object pointing to a file which now contains the same content as the current file.
        """

        def _internal_copy(
            source: "h5py.File",
            source_path: str,
            target: "h5py.File",
            target_path: str,
            maintain_flag: bool,
        ) -> None:
            """
            Internal function to copy content of one HDF5 file to another or copy a group within the same HDF5 file.

            Args:
                source (h5py.File): The source HDF5 File object.
                source_path (str): The path inside the source HDF5 file.
                target (h5py.File): The target HDF5 File object.
                target_path (str): The path inside the target HDF5 file.
                maintain_flag (bool): Whether to maintain the same group name.
            """
            if maintain_flag:
                try:
                    target.create_group(target_path)
                except ValueError:
                    pass  # In case the copy_to() function failed previously and the group already exists.

            if target_path == "/":
                (
                    source.copy(target_path, "/")
                    if source == target
                    else source.copy(target_path, target)
                )
            else:
                if maintain_flag:
                    if dest_path != "":
                        source.copy(source_path, target[dest_path])
                    else:
                        source.copy(source_path, target)
                else:
                    group_name_old = source_path.split("/")[-1]
                    try:
                        target.create_group("/tmp")
                    except ValueError:
                        pass
                    source.copy(source_path, target["/tmp"])
                    try:
                        target.move("/tmp/" + group_name_old, target_path)
                    except ValueError:
                        del target[dest_path]
                        target.move("/tmp/" + group_name_old, target_path)
                    del target["/tmp"]

        if file_name is None:
            file_name = destination.file_name

        if self.file_exists:
            dest_path = (
                destination.h5_path[1:]
                if destination.h5_path[0] == "/"
                else destination.h5_path
            )
            if self.file_name != file_name:
                with _open_hdf(self.file_name, mode="r") as f_source:
                    with _open_hdf(file_name, mode="a") as f_target:
                        _internal_copy(
                            source=f_source,
                            source_path=self._h5_path,
                            target=f_target,
                            target_path=dest_path,
                            maintain_flag=maintain_name,
                        )
            else:
                with _open_hdf(file_name, mode="a") as f_target:
                    _internal_copy(
                        source=f_target,
                        source_path=self._h5_path,
                        target=f_target,
                        target_path=dest_path,
                        maintain_flag=maintain_name,
                    )

        return destination

    def file_size(self) -> float:
        """
        Get the size of the HDF5 file.

        Returns:
            float: The file size in bytes.
        """
        try:
            return os.path.getsize(self.file_name)
        except FileNotFoundError:
            return 0

    def list_all(self) -> List[str]:
        """
        List all groups and nodes in the HDF5 file at the current h5 path.

        Returns:
            List[str]: A combined list of all groups and nodes in the HDF5 file.
        """
        list_dict = self.list_h5_path(h5_path=self.h5_path)
        return list_dict["nodes"] + list_dict["groups"]

    def list_h5_path(self, h5_path: str = "") -> Dict[str, List[str]]:
        """
        List all groups and nodes of the HDF5 file.

        Args:
            h5_path (str, optional): The path to a group in the HDF5 file from where the data is read. Defaults to "".

        Returns:
            Dict[str, List[str]]: A dictionary with keys "groups" and "nodes" containing lists of groups and nodes.
        """
        if h5_path == "":
            h5_path_select = self.h5_path
        elif h5_path[0] == "/":
            h5_path_select = h5_path
        else:
            h5_path_select = posixpath.join(self.h5_path, h5_path)
        if self.file_exists:
            nodes_path_lst, groups_path_lst = list_hdf(
                file_name=self.file_name, h5_path=h5_path_select, recursive=False
            )
            return {
                "groups": [g.rsplit("/", 1)[-1] for g in groups_path_lst],
                "nodes": [n.rsplit("/", 1)[-1] for n in nodes_path_lst],
            }
        else:
            return {"groups": [], "nodes": []}

    def to_dict(self, hierarchical: bool = False) -> Dict[str, Any]:
        """
        Get the content of the HDF5 file at the current h5_path returned as a dictionary.

        Args:
            hierarchical (bool, optional): Whether to convert the internal hierarchy of the HDF5 file to a hierarchical dictionary. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary with the content of the HDF5 file.
        """
        try:
            with h5py.File(self._file_name, "r") as hdf:
                path_dict = _read_dict_from_open_hdf(
                    hdf_filehandle=hdf,
                    h5_path=self._h5_path,
                    recursive=True,
                    slash="ignore",
                )
        except (FileNotFoundError, KeyError):
            return {}
        else:
            if self._h5_path == "/":
                replace_path = "/"
            else:
                replace_path = self._h5_path + "/"
            rel_path_dict = {
                k.replace(replace_path, "", 1): v for k, v in path_dict.items()
            }
            if hierarchical:
                return get_hierarchical_dict(path_dict=rel_path_dict)
            else:
                return rel_path_dict

    def write_dict(self, data_dict: Dict[str, Any], compression: int = 4) -> None:
        """
        Write a dictionary to the HDF5 file.

        Args:
            data_dict (Dict[str, Any]): Dictionary of data objects to be stored in the HDF5 file, the keys provide the
                                        path inside the HDF5 file and the values the data to be stored in those nodes.
                                        The corresponding HDF5 groups are created automatically:
                                            {
                                                '/hdf5root/group/node_name': {},
                                                '/hdf5root/group/subgroup/node_name': [...],
                                            }
            compression (int, optional): The compression level to use (0-9) to compress data using gzip. Defaults to 4.
        """

        write_dict_to_hdf(
            file_name=self.file_name,
            data_dict={
                posixpath.join(self.h5_path, k): v for k, v in data_dict.items()
            },
            compression=compression,
        )

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Represent the Pointer inside an interactive python shell or Jupyter Notebooks. In particular in Jupyter lab
        the content of the HDF5 file can be browsed interactively. This function recursively loads all the content below
        the current h5_path.

        Returns:
            Dict[str, Any]: A dictionary of the hierarchy of the HDF5 file at the current h5_path.
        """
        return convert_dict_items_to_str(input_dict=self.to_dict(hierarchical=True))

    def __delitem__(self, key: str) -> None:
        """
        Delete an item from the HDF5 file.

        Args:
            key (str): The key of the item to delete.
        """
        if self.file_exists:
            delete_item(
                file_name=self.file_name, h5_path=posixpath.join(self._h5_path, key)
            )

    def __getitem__(self, item: Union[str, slice]) -> Union[Dict, List, float, int]:
        """
        Get/read data from the HDF5 file.

        Args:
            item (Union[str, slice]): The path to the data or the key of the data object.

        Returns:
            Union[Dict, List, float, int]: The data or data object.
        """
        if self._h5_path != "/":
            h5_path_new = self._h5_path + "/" + item
        else:
            h5_path_new = self._h5_path + item
        try:
            with h5py.File(self._file_name, "r") as hdf:
                data_dict = _read_dict_from_open_hdf(
                    hdf_filehandle=hdf,
                    h5_path=h5_path_new,
                    recursive=False,
                    slash="ignore",
                )
            if len(data_dict) > 1:
                return get_hierarchical_dict(
                    path_dict={
                        k.replace(self._h5_path + "/", ""): v
                        for k, v in data_dict.items()
                    }
                )
            else:
                return list(data_dict.values())[-1]
        except (FileNotFoundError, ValueError):
            return self.__class__(file_name=self._file_name, h5_path=h5_path_new)

    def __str__(self):
        """
        Convert Pointer to string

        Returns:
            str: String representation of the Pointer object
        """
        return (
            'Pointer(file_name="'
            + str(self.file_name)
            + '", h5_path="'
            + str(self.h5_path)
            + '") '
            + str(self.list_h5_path())
        )

    def __setitem__(self, key, value):
        """
        Store data inside the HDF5 file

        Args:
            key (str): key to store the data
            value (pandas.DataFrame, pandas.Series, dict, list, float, int): basically any kind of data is supported
        """
        write_dict_to_hdf(
            file_name=self.file_name,
            data_dict={posixpath.join(self._h5_path, key): value},
        )

    def __del__(self):
        """
        Delete instance of the H5Pointer object - this does not delete the HDF5 file
        """
        del self._file_name
        del self._h5_path

    def __enter__(self):
        """
        Compatibility function for the with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Compatibility function for the with statement
        """
        return None

    def __len__(self):
        """
        Total number of nodes in the current HDF5 file

        Returns:
            int: Total number of nodes in the current HDF5 file
        """
        return len(self.to_dict())

    def __iter__(self):
        """
        Iterate over all node h5 paths in the HDF5 file

        Returns:
            iterable: Iterator over all node h5 paths in the HDF5
        """
        return iter(self.to_dict())
