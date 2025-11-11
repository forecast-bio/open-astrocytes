"""Open astrocyte dynamics data"""

##
# Imports

from ._datasets import (
    Hive,
    _DEFAULT_HIVE_ROOT,
)


##
# Expose 

hive = Hive( root = _DEFAULT_HIVE_ROOT )