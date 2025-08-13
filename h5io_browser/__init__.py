import h5io_browser._version
from h5io_browser.base import (
    delete_item,
    list_hdf,
    read_dict_from_hdf,
    write_dict_to_hdf,
)
from h5io_browser.base import (
    read_dict_from_hdf as read_nested_dict_from_hdf,
)
from h5io_browser.pointer import Pointer

__version__ = h5io_browser._version.__version__
__all__ = [
    Pointer,
    delete_item,
    list_hdf,
    read_dict_from_hdf,
    read_nested_dict_from_hdf,
    write_dict_to_hdf,
]
