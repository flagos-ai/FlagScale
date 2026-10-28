# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# The three names ms-swift's loss_scale pulls from swift.template, extracted so
# this package stands alone. Vendored from ms-swift (Apache-2.0):
# swift/template/utils.py.

from typing import Any, Dict, List, Union

Message = Dict[str, Union[str, List[Dict[str, Any]], List[int], None]]
Messages = List[Message]


class ContextType:
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    OTHER = 'other'


def get_last_user_round(messages):
    """Get the index of the last user or tool message."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] in ('user', 'tool'):
            return i
    return -1
