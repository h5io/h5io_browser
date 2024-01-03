from h5io_browser.pointer import Pointer # fmt: skip
from h5io_browser.base import ( # fmt: skip
    delete_item,
    list_hdf,
    read_dict_from_hdf,
    write_dict_to_hdf,
)

from ._version import get_versions

# Set version of h5io_browser
__version__ = get_versions()["version"]
