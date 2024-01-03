import os
from collections.abc import MutableMapping
import posixpath
from h5io_browser.base import (
    delete_item,
    list_hdf,
    _open_hdf,
    read_dict_from_hdf,
    write_dict_to_hdf,
)


def convert_dict_items_to_str(input_dict):
    """
    Recursively convert all kys and values to strings using the str() method.

    Args:
        input_dict (dict): dictionary to be converted, can be hierarchical

    Returns:
        dict: dictionary with all keys being strings and all values being either dictionaries or also strings on all
              levels of the hierarchy.
    """
    return {
        str(k): convert_dict_items_to_str(input_dict=v)
        if isinstance(v, dict)
        else str(v)
        for k, v in input_dict.items()
    }


def get_hierarchical_dict(path_dict):
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
    def __init__(self, file_name, h5_path="/"):
        file_name += ".h5" if not file_name.endswith(".h5") else ""
        self._file_name = None
        self._h5_path = None
        self.file_name = file_name
        self.h5_path = h5_path

    @property
    def file_exists(self):
        """
        Check if the HDF5 file exists already

        Returns:
            bool: [True/False]
        """
        if os.path.isfile(self.file_name):
            return True
        else:
            return False

    @property
    def file_name(self):
        """
        Get the file name of the HDF5 file

        Returns:
            str: absolute path to the HDF5 file
        """
        return self._file_name

    @file_name.setter
    def file_name(self, new_file_name):
        """
        Set the file name of the HDF5 file

        Args:
            new_file_name (str): absolute path to the HDF5 file
        """
        self._file_name = os.path.abspath(new_file_name).replace("\\", "/")

    @property
    def h5_path(self):
        """
        Get the path in the HDF5 file starting from the root group - meaning this path starts with '/'

        Returns:
            str: HDF5 path
        """
        return self._h5_path

    @h5_path.setter
    def h5_path(self, path):
        """
        Set the path in the HDF5 file starting from the root group

        Args:
            path (str): HDF5 path
        """
        if (path is None) or (path == ""):
            path = "/"
        self._h5_path = posixpath.normpath(path)
        if not posixpath.isabs(self._h5_path):
            self._h5_path = "/" + self._h5_path

    @property
    def is_empty(self):
        """
        Check if the HDF5 file is empty

        Returns:
            bool: [True/False]
        """
        if self.file_exists:
            nodes, groups = list_hdf(
                file_name=self.file_name, h5_path="/", recursive=True
            )
            return (len(nodes) + len(groups)) == 0
        else:
            return True

    @property
    def is_root(self):
        """
        Check if the current h5_path is pointing to the HDF5 root group.

        Returns:
            bool: [True/False]
        """
        return "/" == self.h5_path

    def copy_to(self, destination, file_name=None, maintain_name=True):
        """
        Copy the content of the HDF5 file to a new location

        Args:
            destination (FileHDFio): FileHDFio object pointing to the new location
            file_name (str): name of the new HDF5 file - optional
            maintain_name (bool): by default the names of the HDF5 groups are maintained

        Returns:
            FileHDFio: FileHDFio object pointing to a file which now contains the same content as file of the current
                       FileHDFio object.
        """

        def _internal_copy(source, source_path, target, target_path, maintain_flag):
            """
            Internal function to copy content of one HDF5 file to another or copy a group within the same HDF5 file.

            Args:
                source (h5py.File): HDF5 File object
                source_path (str): Path inside the source HDF5 file
                target (h5py.File): HDF5 File object
                target_path (str): Path inside the target HDF5 file
                maintain_flag (bool): Maintain the same group name
            """
            if maintain_flag:
                try:
                    target.create_group(target_path)
                except ValueError:
                    pass  # In case the copy_to() function failed previously and the group already exists.

            if target_path == "/":
                source.copy(target_path, "/") if source == target else source.copy(
                    target_path, target
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

    def file_size(self):
        """
        Get size of the HDF5 file

        Returns:
            float: file size in Bytes
        """
        try:
            return os.path.getsize(self.file_name)
        except FileNotFoundError:
            return 0

    def list_all(self):
        """
        List all groups and nodes in the HDF5 file as the current h5 path

        Returns:
            list: combined list of all groups and nodes in the HDF5 file
        """
        list_dict = self.list_h5_path(h5_path=self.h5_path)
        return list_dict["nodes"] + list_dict["groups"]

    def list_h5_path(self, h5_path=""):
        """
        List all groups and nodes of the HDF5 file - where groups are equivalent to directories and nodes to files.

        Args:
            h5_path (str): Path to a group in the HDF5 file from where the data is read

        Returns:
            dict: {'groups': [list of groups], 'nodes': [list of nodes]}
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

    def to_dict(self, hierarchical=False):
        """
        Get the content of the HDF5 file at the current h5_path returned as a dictionary. This includes all sub-groups
        and nodes on all levels of the hierarchy of the HDF5 file below the selected h5_path.

        Args:
            hierarchical (bool): If this parameter is false the returned dictionary has the h5_paths as keys and the
                                 corresponding nodes as values. If this parameter is true, then the internal hierarchy
                                 of the HDF5 file is converted to an hierarchical dictionary.

        Returns:
            dict: Dictionary with the content of the HDF5 file
        """
        try:
            path_dict = read_dict_from_hdf(
                file_name=self._file_name,
                h5_path=self._h5_path,
                recursive=True,
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

    def write_dict(self, data_dict, compression=4):
        """
        Write dictionary to HDF5 file

        Args:
            data_dict (dict): Dictionary of data objects to be stored in the HDF5 file, the keys provide the path inside
                              the HDF5 file and the values the data to be stored in those nodes. The corresponding HDF5
                              groups are created automatically:
                                  {
                                      '/hdf5root/group/node_name': {},
                                      '/hdf5root/group/subgroup/node_name': [...],
                                  }
            compression (int): Compression level to use (0-9) to compress data using gzip.
        """
        write_dict_to_hdf(
            file_name=self.file_name,
            data_dict={
                posixpath.join(self.h5_path, k): v for k, v in data_dict.items()
            },
            compression=compression,
        )

    def _repr_json_(self):
        """
        Represent the Pointer inside an interactive python shell or Jupyter Notebooks. In particular in Jupyter lab
        the content of the HDF5 file can be browsed interactively. This function recursively loads all the content below
        the current h5_path.

        Returns:
            dict: Dictionary of the hierarchy of the HDF5 file at the current h5_path
        """
        return convert_dict_items_to_str(input_dict=self.to_dict(hierarchical=True))

    def __delitem__(self, key):
        """
        Delete item from the HDF5 file

        Args:
            key (str): key of the item to delete
        """
        if self.file_exists:
            delete_item(
                file_name=self.file_name, h5_path=posixpath.join(self._h5_path, key)
            )

    def __getitem__(self, item):
        """
        Get/ read data from the HDF5 file

        Args:
            item (str, slice): path to the data or key of the data object

        Returns:
            dict, list, float, int: data or data object
        """
        if self._h5_path != "/":
            h5_path_new = self._h5_path + "/" + item
        else:
            h5_path_new = self._h5_path + item
        try:
            data_dict = read_dict_from_hdf(
                file_name=self._file_name,
                h5_path=h5_path_new,
                recursive=False,
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
