import os
import numpy as np
import h5py
from unittest import TestCase
import posixpath
import h5io
import h5py
from h5io_browser import (
    delete_item,
    list_hdf,
    read_dict_from_hdf,
    write_dict_to_hdf,
)
from h5io_browser.base import (
    _get_hdf_content,
    _is_ragged_in_1st_dim_only,
    _read_dict_from_open_hdf,
    _read_hdf,
    _write_hdf,
)


def _read_dict_from_hdf(
    file_name: str, h5_path: str, recursive: bool = False, slash: str = "ignore"
) -> dict:
    """
    Read data from HDF5 file into a dictionary - by default only the nodes are converted to dictionaries, additional
    sub groups can be converted using the recursive parameter.

    Args:
       file_name (str): Name of the file on disk
       h5_path (str): Path to a group in the HDF5 file from where the data is read
       recursive (bool/int): Recursively browse through the HDF5 file, either a boolean flag or an integer
                              which specifies the level of recursion.
       slash (str): 'ignore' | 'replace' Whether to replace the string {FWDSLASH} with the value /. This does
                    not apply to the top level name (title). If 'ignore', nothing will be replaced.
    Returns:
       dict: The loaded data as dictionary, with the keys being the path inside the HDF5 file. The values can be of
             any type supported by ``write_hdf5``.
    """
    with h5py.File(file_name, "r") as hdf:
        return _read_dict_from_open_hdf(
            hdf_filehandle=hdf, h5_path=h5_path, recursive=recursive, slash=slash
        )


def get_hdf5_raw_content(file_name):
    item_lst = []

    def collect_attrs(name, obj):
        item_lst.append({name: {k: v for k, v in obj.attrs.items()}})

    with h5py.File(file_name, "r") as f:
        f.visititems(collect_attrs)

    return item_lst


class TestBaseHelperFunctions(TestCase):
    def test_ragged_array(self):
        """Should correctly identify ragged arrays/lists."""
        self.assertTrue(
            _is_ragged_in_1st_dim_only([[1], [1, 2]]),
            "Ragged nested list not detected!",
        )
        self.assertTrue(
            _is_ragged_in_1st_dim_only([np.array([1]), np.array([1, 2])]),
            "Ragged list of arrays not detected!",
        )
        self.assertFalse(
            _is_ragged_in_1st_dim_only([[1, 2], [3, 4]]),
            "Non-ragged nested list detected incorrectly!",
        )
        self.assertFalse(
            _is_ragged_in_1st_dim_only(np.array([[1, 2], [3, 4]])),
            "Non-ragged array detected incorrectly!",
        )
        self.assertTrue(
            _is_ragged_in_1st_dim_only([[[1]], [[2], [3]]]),
            "Ragged nested list not detected even though shape[1:] matches!",
        )
        self.assertFalse(
            _is_ragged_in_1st_dim_only([[[1, 2, 3]], [[2]], [[3]]]),
            "Ragged nested list detected incorrectly even though shape[1:] don't match!",
        )


class TestBaseHierachical(TestCase):
    def setUp(self):
        self.file_name = "test_hierarchical.h5"
        self.h5_path = "/data_hierarchical"
        self.data_hierarchical = {
            "/data_hierarchical/a": [1, 2],
            "/data_hierarchical/b": 3,
            "/data_hierarchical/c/d": 4,
            "/data_hierarchical/c/e": 5,
        }
        write_dict_to_hdf(file_name=self.file_name, data_dict=self.data_hierarchical)

    def tearDown(self):
        os.remove(self.file_name)

    def test_read_dict_hierarchical(self):
        self.assertEqual(
            self.data_hierarchical,
            _read_dict_from_hdf(
                file_name=self.file_name, h5_path=self.h5_path, recursive=True
            ),
        )
        self.assertEqual(
            self.data_hierarchical,
            _read_dict_from_hdf(
                file_name=self.file_name, h5_path=self.h5_path, recursive=1
            ),
        )
        self.assertEqual({}, _read_dict_from_hdf(file_name=self.file_name, h5_path="/"))
        self.assertEqual(
            {
                k: v
                for k, v in self.data_hierarchical.items()
                if "/data_hierarchical/c" in k
            },
            _read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=posixpath.join(self.h5_path, "c"),
                recursive=True,
            ),
        )
        self.assertEqual(
            {"/data_hierarchical/a": [1, 2]},
            _read_dict_from_hdf(
                file_name=self.file_name, h5_path=posixpath.join(self.h5_path, "a")
            ),
        )
        self.assertEqual(
            {"/data_hierarchical/b": 3},
            _read_dict_from_hdf(
                file_name=self.file_name, h5_path=posixpath.join(self.h5_path, "b")
            ),
        )

    def test_read_nested_dict_hierarchical(self):
        self.assertEqual(
            {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=self.h5_path,
                recursive=True,
            ),
        )
        self.assertEqual(
            {"data_hierarchical": {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}}},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path="/",
                recursive=True,
            ),
        )
        self.assertEqual(
            {"d": 4, "e": 5},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=posixpath.join(self.h5_path, "c"),
                recursive=True,
            ),
        )
        self.assertEqual(
            {"a": [1, 2], "b": 3},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=self.h5_path,
                recursive=False,
            ),
        )
        self.assertEqual(
            {"a": [1, 2]},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=posixpath.join(self.h5_path, "a"),
                recursive=False,
            ),
        )
        self.assertEqual(
            {"b": 3},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=posixpath.join(self.h5_path, "b"),
                recursive=False,
            ),
        )
        self.assertEqual(
            {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=self.h5_path,
                group_paths=[posixpath.join(self.h5_path, "c")],
                recursive=False,
            ),
        )
        self.assertEqual(
            {"data_hierarchical": {"c": {"d": 4, "e": 5}}},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path="/",
                group_paths=[posixpath.join(self.h5_path, "c")],
                recursive=False,
            ),
        )

    def test_read_nested_dict_hierarchical_pattern(self):
        self.assertEqual(
            {"c": {"e": 5}},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=self.h5_path,
                recursive=True,
                pattern="*/e",
            ),
        )
        self.assertEqual(
            {"data_hierarchical": {"c": {"d": 4, "e": 5}}},
            read_dict_from_hdf(
                file_name=self.file_name, h5_path="/", recursive=True, pattern="*/c/*"
            ),
        )
        self.assertEqual(
            {"d": 4},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=posixpath.join(self.h5_path, "c"),
                recursive=True,
                pattern="*/d",
            ),
        )
        self.assertEqual(
            {"b": 3},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=self.h5_path,
                recursive=False,
                pattern="*/b",
            ),
        )
        self.assertEqual(
            {"a": [1, 2]},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=posixpath.join(self.h5_path, "a"),
                recursive=False,
                pattern="a",
            ),
        )
        self.assertEqual(
            {"b": 3},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=posixpath.join(self.h5_path, "b"),
                recursive=False,
                pattern="*/b",
            ),
        )
        self.assertEqual(
            {"c": {"d": 4, "e": 5}},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path=self.h5_path,
                group_paths=[posixpath.join(self.h5_path, "c")],
                recursive=False,
                pattern="*/c/*",
            ),
        )
        self.assertEqual(
            {"data_hierarchical": {"c": {"d": 4, "e": 5}}},
            read_dict_from_hdf(
                file_name=self.file_name,
                h5_path="/",
                group_paths=[posixpath.join(self.h5_path, "c")],
                recursive=False,
                pattern="*/c/*",
            ),
        )

    def test_read_hdf(self):
        self.assertEqual(
            _read_hdf(
                hdf_filehandle=self.file_name,
                h5_path=posixpath.join(self.h5_path, "b"),
                slash="ignore",
            ),
            3,
        )
        with h5py.File(self.file_name, "r") as hdf:
            self.assertEqual(
                _read_hdf(
                    hdf_filehandle=hdf,
                    h5_path=posixpath.join(self.h5_path, "b"),
                    slash="ignore",
                ),
                3,
            )
        with self.assertRaises(TypeError):
            self.assertEqual(
                _read_hdf(
                    hdf_filehandle=1,
                    h5_path=posixpath.join(self.h5_path, "b"),
                    slash="ignore",
                ),
                3,
            )

    def test_hdf5_structure(self):
        self.assertEqual(
            get_hdf5_raw_content(file_name=self.file_name),
            [
                {"data_hierarchical": {}},
                {"data_hierarchical/a": {"TITLE": "json"}},
                {"data_hierarchical/b": {"TITLE": "int"}},
                {"data_hierarchical/c": {}},
                {"data_hierarchical/c/d": {"TITLE": "int"}},
                {"data_hierarchical/c/e": {"TITLE": "int"}},
            ],
        )

    def test_list_hdf(self):
        nodes, groups = list_hdf(file_name=self.file_name, h5_path=self.h5_path)
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/wrong_path")
        self.assertEqual(nodes, [])
        self.assertEqual(groups, [])
        nodes, groups = list_hdf(file_name="empty.h5", h5_path=self.h5_path)
        self.assertEqual(nodes, [])
        self.assertEqual(groups, [])
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/")
        self.assertEqual(groups, ["/data_hierarchical"])
        self.assertEqual(nodes, [])
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/", recursive=1)
        self.assertEqual(groups, ["/data_hierarchical", "/data_hierarchical/c"])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/", recursive=2)
        self.assertEqual(groups, ["/data_hierarchical", "/data_hierarchical/c"])
        self.assertEqual(
            nodes,
            [
                "/data_hierarchical/a",
                "/data_hierarchical/b",
                "/data_hierarchical/c/d",
                "/data_hierarchical/c/e",
            ],
        )
        nodes, groups = list_hdf(
            file_name=self.file_name, h5_path=self.h5_path, recursive=True
        )
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(
            nodes,
            [
                "/data_hierarchical/a",
                "/data_hierarchical/b",
                "/data_hierarchical/c/d",
                "/data_hierarchical/c/e",
            ],
        )
        nodes, groups = list_hdf(
            file_name=self.file_name,
            h5_path=posixpath.join(self.h5_path, "a"),
            recursive=True,
        )
        self.assertEqual(groups, [])
        self.assertEqual(nodes, [])
        with self.assertRaises(TypeError):
            list_hdf(file_name=self.file_name, h5_path="/", recursive=1.0)

    def test_list_hdf_pattern(self):
        nodes, groups = list_hdf(
            file_name=self.file_name, h5_path=self.h5_path, pattern="*/*"
        )
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        nodes, groups = list_hdf(
            file_name=self.file_name, h5_path="/data_hierarchical", pattern="*/d"
        )
        self.assertEqual(nodes, [])
        self.assertEqual(groups, [])
        nodes, groups = list_hdf(
            file_name=self.file_name, h5_path="/", recursive=1, pattern="*/c"
        )
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(nodes, [])
        nodes, groups = list_hdf(
            file_name=self.file_name, h5_path="/", recursive=2, pattern="*/c/*"
        )
        self.assertEqual(groups, [])
        self.assertEqual(
            nodes,
            [
                "/data_hierarchical/c/d",
                "/data_hierarchical/c/e",
            ],
        )

    def test_get_hdf_content(self):
        with h5py.File(self.file_name, "r") as hdf:
            nodes, groups = _get_hdf_content(
                hdf=hdf["data_hierarchical"],
                recursive=False,
                only_groups=False,
                only_nodes=False,
            )
            self.assertEqual(groups, ["/data_hierarchical/c"])
            self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
            nodes = _get_hdf_content(
                hdf=hdf["data_hierarchical"],
                recursive=False,
                only_groups=False,
                only_nodes=True,
            )
            groups = _get_hdf_content(
                hdf=hdf["data_hierarchical"],
                recursive=False,
                only_groups=True,
                only_nodes=False,
            )
            self.assertEqual(groups, ["/data_hierarchical/c"])
            self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
            self.assertEqual(groups, ["/data_hierarchical/c"])
            self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])

    def test_delete(self):
        delete_item(file_name=self.file_name, h5_path=posixpath.join(self.h5_path, "c"))
        nodes, groups = list_hdf(file_name=self.file_name, h5_path=self.h5_path)
        self.assertEqual(groups, [])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        delete_item(file_name=self.file_name, h5_path="/data_hierarchical/a")
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(groups, [])
        self.assertEqual(nodes, ["/data_hierarchical/b"])
        delete_item(file_name=self.file_name, h5_path=posixpath.join(self.h5_path, "d"))
        delete_item(file_name="empty.h5", h5_path=posixpath.join(self.h5_path, "c"))

    def test_write_dict_to_hdf(self):
        nodes, groups = list_hdf(file_name=self.file_name, h5_path=self.h5_path)
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        write_dict_to_hdf(
            file_name=self.file_name,
            data_dict={posixpath.join(self.h5_path, "f"): {"g": 6, "h": 7}},
        )
        write_dict_to_hdf(
            file_name=self.file_name,
            data_dict={posixpath.join(self.h5_path, "i", "l"): 4},
        )
        nodes, groups = list_hdf(file_name=self.file_name, h5_path=self.h5_path)
        self.assertEqual(groups, ["/data_hierarchical/c", "/data_hierarchical/i"])
        self.assertEqual(
            nodes,
            ["/data_hierarchical/a", "/data_hierarchical/b", "/data_hierarchical/f"],
        )
        delete_item(file_name=self.file_name, h5_path=posixpath.join(self.h5_path, "i"))
        delete_item(file_name=self.file_name, h5_path=posixpath.join(self.h5_path, "f"))
        nodes, groups = list_hdf(file_name=self.file_name, h5_path=self.h5_path)
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        write_dict_to_hdf(
            file_name=self.file_name,
            data_dict={posixpath.join(self.h5_path, "j"): ValueError},
        )
        with h5py.File(self.file_name, "r") as hdf:
            self.assertEqual(
                ValueError,
                _read_hdf(
                    hdf_filehandle=hdf,
                    h5_path=posixpath.join(self.h5_path, "j"),
                    slash="ignore",
                ),
            )


class TestBaseJSON(TestCase):
    def setUp(self):
        self.file_name = "test_json.h5"
        self.h5_path = "/data_json"
        self.data_hierarchical = {
            "/data_json/a": [1, 2],
            "/data_json/b": 3,
            "/data_json/c": {"d": 4, "e": 5},
        }
        write_dict_to_hdf(file_name=self.file_name, data_dict=self.data_hierarchical)

    def tearDown(self):
        os.remove(self.file_name)

    def test_read_dict_hierarchical(self):
        self.assertEqual(
            self.data_hierarchical,
            _read_dict_from_hdf(
                file_name=self.file_name, h5_path=self.h5_path, recursive=True
            ),
        )
        self.assertEqual(
            self.data_hierarchical,
            _read_dict_from_hdf(
                file_name=self.file_name, h5_path=self.h5_path, recursive=1
            ),
        )
        self.assertEqual({}, _read_dict_from_hdf(file_name=self.file_name, h5_path="/"))

    def test_hdf5_structure(self):
        self.assertEqual(
            get_hdf5_raw_content(file_name=self.file_name),
            [
                {"data_json": {}},
                {"data_json/a": {"TITLE": "json"}},
                {"data_json/b": {"TITLE": "int"}},
                {"data_json/c": {"TITLE": "json"}},
            ],
        )

    def test_list_groups(self):
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_json")
        self.assertEqual(groups, [])
        self.assertEqual(nodes, ["/data_json/a", "/data_json/b", "/data_json/c"])

    def test_delete(self):
        delete_item(file_name=self.file_name, h5_path="/data_json/c")
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_json")
        self.assertEqual(groups, [])
        self.assertEqual(nodes, ["/data_json/a", "/data_json/b"])
        delete_item(file_name=self.file_name, h5_path="/data_json/b")
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_json")
        self.assertEqual(groups, [])
        self.assertEqual(nodes, ["/data_json/a"])


class TestCompatibility(TestCase):
    def setUp(self):
        self.file_name = "testcomp.h5"
        self.data = {
            "array": np.ones(4) * 42,
            "b": 42,
        }
        self.h5_path = "h5io"
        h5io.write_hdf5("testcomp.h5", self.data)

    def test_h5io(self):
        dataread = h5io.read_hdf5(self.file_name, self.h5_path)
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                self.assertTrue(all(np.equal(v, dataread[k])))
            else:
                self.assertTrue(v == dataread[k])

    def test_read_dict_from_hdf(self):
        dataread = _read_dict_from_hdf(self.file_name, self.h5_path)
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                self.assertTrue(all(np.equal(v, dataread[self.h5_path][k])))
            else:
                self.assertTrue(v == dataread[self.h5_path][k])

    def test_read_nested_dict_from_hdf(self):
        dataread = read_dict_from_hdf(self.file_name, self.h5_path)
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                self.assertTrue(all(np.equal(v, dataread[self.h5_path][k])))
            else:
                self.assertTrue(v == dataread[self.h5_path][k])

    def tearDown(self):
        os.remove(self.file_name)


class TestBasePartialRead(TestCase):
    def setUp(self):
        self.file_name = "test_structured_hdf.h5"
        self.h5_path = "data_hierarchical"
        self.data_hierarchical = {
            "a": np.array([1, 2]),
            "b": 3,
            "c": {"d": np.array([4, 5]), "e": np.array([6, 7])},
        }
        write_dict_to_hdf(
            file_name=self.file_name,
            data_dict={
                posixpath.join(self.h5_path, "a"): self.data_hierarchical["a"],
                posixpath.join(self.h5_path, "b"): self.data_hierarchical["b"],
                posixpath.join(self.h5_path, "c", "d"): self.data_hierarchical["c"][
                    "d"
                ],
                posixpath.join(self.h5_path, "c", "e"): self.data_hierarchical["c"][
                    "e"
                ],
            },
        )

    def tearDown(self):
        os.remove(self.file_name)

    def test_read_dict_hierarchical(self):
        output = read_dict_from_hdf(file_name=self.file_name, h5_path=self.h5_path)
        self.assertTrue(
            np.all(np.equal(output["a"], np.array([1, 2]))),
        )
        self.assertEqual(output["b"], 3)
        output = read_dict_from_hdf(
            file_name=self.file_name,
            h5_path=self.h5_path,
            group_paths=["c"],
        )
        self.assertTrue(
            np.all(np.equal(output["a"], np.array([1, 2]))),
        )
        self.assertEqual(output["b"], 3)
        self.assertTrue(
            np.all(np.equal(output["c"]["d"], np.array([4, 5]))),
        )
        self.assertTrue(
            np.all(np.equal(output["c"]["e"], np.array([6, 7]))),
        )
        output = read_dict_from_hdf(
            file_name=self.file_name,
            h5_path=self.h5_path,
            recursive=True,
        )
        self.assertTrue(
            np.all(np.equal(output["a"], np.array([1, 2]))),
        )
        self.assertEqual(output["b"], 3)
        self.assertTrue(
            np.all(np.equal(output["c"]["d"], np.array([4, 5]))),
        )
        self.assertTrue(
            np.all(np.equal(output["c"]["e"], np.array([6, 7]))),
        )

    def test_write_overwrite_error(self):
        with self.assertRaises(OSError):
            _write_hdf(
                hdf_filehandle=self.file_name,
                data=self.data_hierarchical,
                h5_path=self.h5_path,
                overwrite=False,
            )

    def test_hdf5_structure(self):
        self.assertEqual(
            get_hdf5_raw_content(file_name=self.file_name),
            [
                {"data_hierarchical": {}},
                {"data_hierarchical/a": {"TITLE": "ndarray"}},
                {"data_hierarchical/b": {"TITLE": "int"}},
                {"data_hierarchical/c": {}},
                {"data_hierarchical/c/d": {"TITLE": "ndarray"}},
                {"data_hierarchical/c/e": {"TITLE": "ndarray"}},
            ],
        )
