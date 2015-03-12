from pkg_resources import declare_namespace
declare_namespace(__name__)

import invest
import data
__all__ = ['invest', 'data']