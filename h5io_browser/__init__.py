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

from ._version import get_versions

# Set version of h5io_browser
__version__ = get_versions()["version"]
__all__ = [
    Pointer,
    delete_item,
    list_hdf,
    read_dict_from_hdf,
    read_nested_dict_from_hdf,
    write_dict_to_hdf,
]
