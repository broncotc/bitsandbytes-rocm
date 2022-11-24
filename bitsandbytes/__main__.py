# from bitsandbytes.debug_cli import cli

# cli()
import os
import sys
from warnings import warn

import torch

HEADER_WIDTH = 60


def print_header(
    txt: str, width: int = HEADER_WIDTH, filler: str = "+"
) -> None:
    txt = f" {txt} " if txt else ""
    print(txt.center(width, filler))


def print_debug_info() -> None:
    print(
        "\nAbove we output some debug information. Please provide this info when "
        f"creating an issue via {PACKAGE_GITHUB_URL}/issues/new/choose ...\n"
    )


print_header("")
print_header("DEBUG INFORMATION")
print_header("")
print()


from . import COMPILED_WITH_CUDA, PACKAGE_GITHUB_URL

print_header("")
print_header("DEBUG INFO END")
print_header("")
print(
    """
Running a quick check that:
    + library is importable
    + CUDA function is callable
"""
)

try:
    from bitsandbytes.optim import Adam

    p = torch.nn.Parameter(torch.rand(10, 10).cuda())
    a = torch.rand(10, 10).cuda()

    p1 = p.data.sum().item()

    adam = Adam([p])

    out = a * p
    loss = out.sum()
    loss.backward()
    adam.step()

    p2 = p.data.sum().item()

    assert p1 != p2
    print("SUCCESS!")
    print("Installation was successful!")
    sys.exit(0)

except ImportError:
    print()
    warn(
        f"WARNING: {__package__} is currently running as CPU-only!\n"
        "Therefore, 8-bit optimizers and GPU quantization are unavailable.\n\n"
        f"If you think that this is so erroneously,\nplease report an issue!"
    )
    print_debug_info()
    sys.exit(0)
except Exception as e:
    print(e)
    print_debug_info()
    sys.exit(1)
