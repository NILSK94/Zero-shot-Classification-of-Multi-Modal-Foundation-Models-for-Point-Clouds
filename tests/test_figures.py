from __future__ import annotations

import pandas as pd

from mmfm_tool.figures import confusion_matrix_grid


def test_confusion_matrix_grid_keeps_four_view_panels() -> None:
    rows = pd.DataFrame(
        [
            {
                "dataset": "modelnet",
                "model": "paper-model",
                "status": "ok",
                "view_count": 1,
                "true_label": "chair",
                "pred_label": "chair",
            },
            {
                "dataset": "modelnet",
                "model": "paper-model",
                "status": "ok",
                "view_count": 4,
                "true_label": "chair",
                "pred_label": "table",
            },
        ]
    )

    figure = confusion_matrix_grid(rows, model="paper-model", normalize=True)

    assert len(figure.layout.annotations) >= 4
    assert len(figure.data) == 2
    assert figure.layout.title.text == "paper-model — confusion matrices by number of views (valid responses)"
