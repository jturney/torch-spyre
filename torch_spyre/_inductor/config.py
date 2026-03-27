import os
import sys
from .errors import Unsupported
from torch.utils._config_module import install_config_module

lx_planning: bool = os.environ.get("LX_PLANNING", "0") == "1"

dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))

sencores: int = int(os.getenv("SENCORES", "32"))
if sencores > 32 or sencores < 1:
        raise Unsupported(f"invalid SENCORES value {sencores}")

install_config_module(sys.modules[__name__])
