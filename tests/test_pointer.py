import os
import h5py
from unittest import TestCase
from h5io_browser import Pointer


def get_hdf5_raw_content(file_name):
    item_lst = []

    def collect_attrs(name, obj):
        item_lst.append({name: {k: v for k, v in obj.attrs.items()}})

    with h5py.File(file_name, "r") as f:
        f.visititems(collect_attrs)

    return item_lst


class TestPointer(TestCase):
    def setUp(self):
        self.file_name = "test_hierarchical.h5"
        self.h5_path = "/data_hierarchical"
        self.data_hierarchical = {
            "/data_hierarchical/a": [1, 2],
            "/data_hierarchical/b": 3,
            "/data_hierarchical/c/d": 4,
            "/data_hierarchical/c/e": 5,
        }
        Pointer(file_name=self.file_name, h5_path=self.h5_path).write_dict(
            data_dict=self.data_hierarchical
        )

    def tearDown(self):
        os.remove(self.file_name)

    def test_empty_file_root(self):
        file_name = "empty.h5"
        p = Pointer(file_name=file_name, h5_path="/")
        self.assertTrue(p.is_empty)
        self.assertFalse(p.file_exists)
        self.assertEqual(p.file_name, os.path.abspath(file_name).replace("\\", "/"))
        self.assertEqual(p["test"].h5_path, "/test")
        self.assertEqual(
            p.list_h5_path(h5_path="wrong/path"), {"nodes": [], "groups": []}
        )
        self.assertEqual(
            p.list_h5_path(h5_path="/wrong/path"), {"nodes": [], "groups": []}
        )
        self.assertEqual(p.file_size(), 0)
        self.assertEqual(p.to_dict(hierarchical=True), {})
        self.assertEqual(p.to_dict(hierarchical=False), {})

    def test_h5_path(self):
        file_name = "empty.h5"
        p = Pointer(file_name=file_name, h5_path="/test")
        self.assertEqual(p.h5_path, "/test")
        p = Pointer(file_name=file_name, h5_path="test")
        self.assertEqual(p.h5_path, "/test")
        p.h5_path = None
        self.assertEqual(p.h5_path, "/")
        p.h5_path = ""
        self.assertEqual(p.h5_path, "/")

    def test_is_root(self):
        self.assertTrue(Pointer(file_name="empty.h5", h5_path="/").is_root)
        self.assertTrue(Pointer(file_name="self.file_name", h5_path="/").is_root)
        self.assertFalse(Pointer(file_name="empty.h5", h5_path="/wrong/path").is_root)
        self.assertFalse(
            Pointer(file_name="self.file_name", h5_path="/data_hierarchical").is_root
        )
        self.assertFalse(
            Pointer(file_name="self.file_name", h5_path="/wrong/path").is_root
        )

    def test_navigate_hierarchical_file(self):
        p = Pointer(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertFalse(p.is_empty)
        self.assertTrue(p.file_exists)
        self.assertEqual(p.file_size(), 6182)
        self.assertEqual(p["a"], self.data_hierarchical["/data_hierarchical/a"])
        self.assertEqual(
            p["c"].h5_path,
            Pointer(file_name=self.file_name, h5_path="/data_hierarchical/c").h5_path,
        )
        self.assertEqual(p["c/d"], self.data_hierarchical["/data_hierarchical/c/d"])
        self.assertEqual(len(p), 4)

    def test_to_dict(self):
        p = Pointer(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(
            p.to_dict(hierarchical=True), {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}}
        )
        self.assertEqual(
            p.to_dict(hierarchical=False),
            {
                k.replace("/data_hierarchical/", ""): v
                for k, v in self.data_hierarchical.items()
            },
        )
        p = Pointer(file_name=self.file_name, h5_path="/")
        self.assertEqual(
            p.to_dict(hierarchical=True),
            {"data_hierarchical": {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}}},
        )
        self.assertEqual(
            p.to_dict(hierarchical=False),
            {k.replace("/", "", 1): v for k, v in self.data_hierarchical.items()},
        )
        self.assertEqual(
            str(p),
            'Pointer(file_name="'
            + os.path.abspath(self.file_name).replace("\\", "/")
            + "\", h5_path=\"/\") {'groups': ['data_hierarchical'], 'nodes': []}",
        )

    def test_repr_json(self):
        p = Pointer(file_name=self.file_name, h5_path="/data_hierarchical")
        self.assertEqual(
            p._repr_json_(), {"a": "[1, 2]", "b": "3", "c": {"d": "4", "e": "5"}}
        )

    def test_write_to_file(self):
        p = Pointer(file_name=self.file_name, h5_path="/data_hierarchical")
        p["f/g/h"] = 5
        self.assertEqual(p["f/g/h"], 5)
        self.assertEqual(len(p), 5)
        self.assertEqual(p.list_all(), ["a", "b", "c", "f"])
        del p["f/g/h"]
        self.assertEqual(len(p), 4)
        self.assertEqual(p.list_all(), ["a", "b", "c", "f"])
        del p["f"]
        self.assertEqual(len(p), 4)
        self.assertEqual(p.list_all(), ["a", "b", "c"])

    def test_with_statement(self):
        with Pointer(file_name=self.file_name)["data_hierarchical"] as p:
            self.assertEqual(
                p.to_dict(hierarchical=True),
                {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}},
            )

    def test_iter(self):
        p = Pointer(file_name=self.file_name, h5_path="/data_hierarchical")
        for i, j in zip(p, ["a", "b", "c/d", "c/e"]):
            self.assertEqual(i, j)

    def test_copy_to(self):
        file_name_new = "test_hierarchical_copy.h5"
        p_old = Pointer(file_name=self.file_name, h5_path="/data_hierarchical")
        p_new = Pointer(file_name=file_name_new, h5_path="/")
        p_old.copy_to(destination=p_new)
        self.assertEqual(
            p_old.to_dict(hierarchical=True),
            {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}},
        )
        self.assertEqual(
            p_new.to_dict(hierarchical=True),
            {"data_hierarchical": {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}}},
        )
        os.remove(file_name_new)
        p_old = Pointer(file_name=self.file_name, h5_path="/data_hierarchical")
        p_new = Pointer(file_name=self.file_name, h5_path="/")
        p_old.copy_to(destination=p_new, file_name=file_name_new)
        self.assertEqual(
            p_old.to_dict(hierarchical=True),
            {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}},
        )
        self.assertEqual(
            p_new.to_dict(hierarchical=True),
            {"data_hierarchical": {"a": [1, 2], "b": 3, "c": {"d": 4, "e": 5}}},
        )
        os.remove(file_name_new)
        p_old = Pointer(file_name=self.file_name, h5_path="/data_hierarchical/c")
        p_new = Pointer(file_name=self.file_name, h5_path="/data_hierarchical/f")
        p_old.copy_to(destination=p_new)
        self.assertEqual(p_old.to_dict(hierarchical=True), {"d": 4, "e": 5})
        self.assertEqual(p_new.to_dict(hierarchical=True), {"c": {"d": 4, "e": 5}})
        p_new = Pointer(file_name=self.file_name, h5_path="/data_hierarchical/g")
        p_old.copy_to(destination=p_new, maintain_name=False)
        self.assertEqual(p_old.to_dict(hierarchical=True), {"d": 4, "e": 5})
        self.assertEqual(p_new.to_dict(hierarchical=True), {"d": 4, "e": 5})
