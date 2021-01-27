"""Main configuration file, loads configs/config.yaml."""

import os.path as op
import yaml
from jinja2 import Environment

from box import Box

with open(
    op.join(
        op.abspath(op.join(__file__, op.pardir, op.pardir)),
        "configs/config.yaml",
    ),
    "r",
) as fp:
    raw_config = yaml.full_load(fp)
    config = Environment().from_string(str(raw_config)).render(raw_config)
cfg = Box(yaml.safe_load(config))
