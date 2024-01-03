import os
import numpy as np
import h5py
from unittest import TestCase
from h5io_browser import (
    delete_item,
    list_hdf,
    read_dict_from_hdf,
    write_dict_to_hdf,
)
from h5io_browser.base import _get_hdf_content, _is_ragged_in_1st_dim_only, _read_hdf


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
            read_dict_from_hdf(
                file_name=self.file_name, h5_path=self.h5_path, recursive=True
            ),
        )
        self.assertEqual(
            self.data_hierarchical,
            read_dict_from_hdf(
                file_name=self.file_name, h5_path=self.h5_path, recursive=1
            ),
        )
        self.assertEqual({}, read_dict_from_hdf(file_name=self.file_name, h5_path="/"))
        self.assertEqual(
            {
                k: v
                for k, v in self.data_hierarchical.items()
                if "/data_hierarchical/c" in k
            },
            read_dict_from_hdf(
                file_name=self.file_name, h5_path="/data_hierarchical/c", recursive=True
            ),
        )
        self.assertEqual(
            {"/data_hierarchical/a": [1, 2]},
            read_dict_from_hdf(
                file_name=self.file_name, h5_path="/data_hierarchical/a"
            ),
        )
        self.assertEqual(
            {"/data_hierarchical/b": 3},
            read_dict_from_hdf(
                file_name=self.file_name, h5_path="/data_hierarchical/b"
            ),
        )

    def test_read_hdf(self):
        self.assertEqual(
            _read_hdf(
                hdf_filehandle=self.file_name,
                h5_path="/data_hierarchical/b",
                slash="ignore",
            ),
            3,
        )
        with h5py.File(self.file_name, "r") as hdf:
            self.assertEqual(
                _read_hdf(
                    hdf_filehandle=hdf, h5_path="/data_hierarchical/b", slash="ignore"
                ),
                3,
            )
        with self.assertRaises(TypeError):
            self.assertEqual(
                _read_hdf(
                    hdf_filehandle=1, h5_path="/data_hierarchical/b", slash="ignore"
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

    def test_list_groups(self):
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/wrong_path")
        self.assertEqual(nodes, [])
        self.assertEqual(groups, [])
        nodes, groups = list_hdf(file_name="empty.h5", h5_path="/data_hierarchical")
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
            file_name=self.file_name, h5_path="/data_hierarchical", recursive=True
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
            file_name=self.file_name, h5_path="/data_hierarchical/a", recursive=True
        )
        self.assertEqual(groups, [])
        self.assertEqual(nodes, [])
        with self.assertRaises(TypeError):
            list_hdf(file_name=self.file_name, h5_path="/", recursive=1.0)

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
        delete_item(file_name=self.file_name, h5_path="/data_hierarchical/c")
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(groups, [])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        delete_item(file_name=self.file_name, h5_path="/data_hierarchical/a")
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(groups, [])
        self.assertEqual(nodes, ["/data_hierarchical/b"])
        delete_item(file_name=self.file_name, h5_path="/data_hierarchical/d")
        delete_item(file_name="empty.h5", h5_path="/data_hierarchical/c")

    def test_write_dict_to_hdf(self):
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        write_dict_to_hdf(
            file_name=self.file_name,
            data_dict={"/data_hierarchical/f": {"g": 6, "h": 7}},
        )
        write_dict_to_hdf(
            file_name=self.file_name, data_dict={"/data_hierarchical/i/l": 4}
        )
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(groups, ["/data_hierarchical/c", "/data_hierarchical/i"])
        self.assertEqual(
            nodes,
            ["/data_hierarchical/a", "/data_hierarchical/b", "/data_hierarchical/f"],
        )
        delete_item(file_name=self.file_name, h5_path="/data_hierarchical/i")
        delete_item(file_name=self.file_name, h5_path="/data_hierarchical/f")
        nodes, groups = list_hdf(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(groups, ["/data_hierarchical/c"])
        self.assertEqual(nodes, ["/data_hierarchical/a", "/data_hierarchical/b"])
        with self.assertRaises(TypeError):
            write_dict_to_hdf(
                file_name=self.file_name, data_dict={"/data_hierarchical/j": ValueError}
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
            read_dict_from_hdf(
                file_name=self.file_name, h5_path=self.h5_path, recursive=True
            ),
        )
        self.assertEqual(
            self.data_hierarchical,
            read_dict_from_hdf(
                file_name=self.file_name, h5_path=self.h5_path, recursive=1
            ),
        )
        self.assertEqual({}, read_dict_from_hdf(file_name=self.file_name, h5_path="/"))

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
