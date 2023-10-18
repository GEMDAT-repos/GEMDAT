from __future__ import annotations

from functools import partial

from matplotlib.testing.decorators import image_comparison

image_comparison2 = partial(
    image_comparison,
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
