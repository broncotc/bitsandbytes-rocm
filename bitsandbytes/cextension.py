import ctypes as ct
import torch

from pathlib import Path
from warnings import warn


class CUDASetup(object):
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def generate_instructions(self):
        self.add_log_entry('CUDA SETUP: Something unexpected happened. Please compile from source:')
        self.add_log_entry('git clone git@github.com:TimDettmers/bitsandbytes.git')
        self.add_log_entry('cd bitsandbytes')
        self.add_log_entry("<make_cmd here, commented out>")
        self.add_log_entry('python setup.py install')

    def initialize(self):
        self.has_printed = False
        self.lib = None
        self.run_cuda_setup()

    def run_cuda_setup(self):
        self.initialized = True
        self.cuda_setup_log = []

        binary_name = "libbitsandbytes_hip.so"
        package_dir = Path(__file__).parent
        binary_path = package_dir / binary_name

        try:
            if not binary_path.exists():
                raise Exception('CUDA SETUP: Setup Failed!')
            else:
                self.add_log_entry(f"CUDA SETUP: Loading binary {binary_path}...")
                self.lib = ct.cdll.LoadLibrary(binary_path)
        except Exception as ex:
            self.add_log_entry(str(ex))
            self.print_log_stack()

    def add_log_entry(self, msg, is_warning=False):
        self.cuda_setup_log.append((msg, is_warning))

    def print_log_stack(self):
        for msg, is_warning in self.cuda_setup_log:
            if is_warning:
                warn(msg)
            else:
                print(msg)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


lib = CUDASetup.get_instance().lib
try:
    if lib is None and torch.cuda.is_available():
        CUDASetup.get_instance().generate_instructions()
        CUDASetup.get_instance().print_log_stack()
        raise RuntimeError('''
        CUDA Setup failed despite GPU being available. Inspect the CUDA SETUP outputs aboveto fix your environment!
        If you cannot find any issues and suspect a bug, please open an issue with detals about your environment:
        https://github.com/TimDettmers/bitsandbytes/issues''')
    lib.cadam32bit_g32
    lib.get_context.restype = ct.c_void_p
    lib.get_cusparse.restype = ct.c_void_p
    COMPILED_WITH_CUDA = True
except AttributeError:
    warn("The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers and GPU quantization are unavailable.")
    COMPILED_WITH_CUDA = False
