"""Give UniDepth the xformers NystromAttention its decoder imports.

xformers 0.0.29 removed components/attention/nystrom.py, and UniDepth still
imports NystromAttention from there, so the xformers that matches Torch 2.7.1
cannot load it. xformers_nystrom.py beside this file is that module unchanged
from xformers 0.0.28.post1 (BSD licence, XFORMERS_LICENSE); everything it
imports still ships in newer xformers. Only the processes that run UniDepth put
this directory on PYTHONPATH, so Python runs this at their start-up.
"""

try:
    import xformers.components.attention as _attention
except ImportError:
    _attention = None

if _attention is not None and not hasattr(_attention, "NystromAttention"):
    from xformers_nystrom import NystromAttention

    _attention.NystromAttention = NystromAttention
