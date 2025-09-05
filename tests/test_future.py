import os
import unittest
from concurrent.futures import Future
from h5io_browser import write_dict_to_hdf, read_future_dict_from_hdf


class TestFuture(unittest.TestCase):
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

    def test_read_future_dict_from_hdf(self):
        data = read_future_dict_from_hdf(
            file_name=self.file_name, h5_path=self.h5_path, recursive=True
        )
        self.assertIsInstance(data, dict)
        self.assertEqual(set(data.keys()), {"a", "b", "c/d", "c/e"})
        for v in data.values():
            self.assertIsInstance(v, Future)
            self.assertTrue(v.done())
        self.assertEqual(data["a"].result(), [1, 2])
        self.assertEqual(data["b"].result(), 3)
        self.assertEqual(data["c/d"].result(), 4)
        self.assertEqual(data["c/e"].result(), 5)
