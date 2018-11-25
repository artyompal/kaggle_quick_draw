""" Data loaders for training & validation. """

from typing import Any

import data_loader_v1
import data_loader_v2
import data_loader_v2a_white
import data_loader_v3
import data_loader_v4_raw
import data_loader_v4a_time
import data_loader_v4b_country
import data_loader_v4e_gtime
import data_loader_v5_6channels
import data_loader_v6_3channels

def get_data_loader(name: str, root: str, transform: Any, num_classes: int,
                    mode: str, image_size: int) -> Any:
    if name == "data_loader_v1":
        return data_loader_v1.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v2":
        return data_loader_v2.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v2a_white":
        return data_loader_v2a_white.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v3":
        return data_loader_v3.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v4_raw":
        return data_loader_v4_raw.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v4a_time":
        return data_loader_v4a_time.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v4b_country":
        return data_loader_v4b_country.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v4e_gtime":
        return data_loader_v4e_gtime.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v5_6channels":
        return data_loader_v5_6channels.DatasetFolder(root, transform, num_classes, mode, image_size)
    elif name == "data_loader_v6_3channels":
        return data_loader_v6_3channels.DatasetFolder(root, transform, num_classes, mode, image_size)
    else:
        assert False
