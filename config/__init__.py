from yacs.config import CfgNode

from .default import get_cfg_defaults  # noqa


_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}


def _valid_type(value, allow_cfg_node=False):
    return (type(value) in _VALID_TYPES) or (
        allow_cfg_node and isinstance(value, CfgNode)
    )


def convert_CN_to_dict(cfg_node, key_list):
    if not isinstance(cfg_node, CfgNode):
        assert _valid_type(cfg_node), "Key {} with value {} is not a valid type; valid types: {}".format(
                                      ".".join(key_list), type(cfg_node), _VALID_TYPES)
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_CN_to_dict(v, key_list + [k])
        return cfg_dict
