from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path
from typing import Any

from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import image_comparison

image_comparison2 = partial(
    image_comparison,
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)


def assert_figures_similar(fig, *, name: str, ext: str = 'png', rms: float = 0.0):
    """Compare plotly figures and raise if different."""
    # Ensure same font is used on different machines (local/CI)
    fig.update_layout(
        font_family='Arial',
        title_font_family='Arial',
    )

    # Get path of caller
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    modulepath = Path(module.__file__)  # type: ignore

    results_dir = Path() / 'result_images' / modulepath.stem
    results_dir.mkdir(exist_ok=True, parents=True)

    filename = f'{name}.{ext}'

    actual = results_dir / filename
    fig.write_image(actual)

    expected_dir = modulepath.parent / 'baseline_images' / modulepath.stem
    expected = expected_dir / filename
    expected_link = results_dir / f'{name}-expected.{ext}'

    if expected_link.exists():
        expected_link.unlink()

    expected_link.symlink_to(expected)

    err: dict[str, Any] = compare_images(
        expected=str(expected_link), actual=str(actual), tol=rms, in_decorator=True
    )  # type: ignore

    if err:
        for key in ('actual', 'expected', 'diff'):
            err[key] = Path(err[key]).relative_to('.')
        raise AssertionError(
            (
                'images not close (RMS {rms:.3f}):'
                '\n\t{actual}\n\t{expected}\n\t{diff}'.format(**err)
            )
        )
