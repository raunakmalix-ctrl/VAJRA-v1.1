"""
eval — objective quality measurement for finished edits.

Turns "does it sound fake?" into numbers, and grounds those numbers in the
recording itself rather than in absolute thresholds: an edit's splice is judged
against the natural variation across genuine word boundaries nearby, because
that is the variation a listener is already accustomed to in that voice and room.

Metrics that need a model (speaker similarity, MOS, independent-ASR
intelligibility, SyncNet lip-sync) are named in `pending_metrics()` rather than
approximated, so a report is explicit about what it did not measure.
"""
from evaluation.scorecard import (
    natural_seam_distribution, seam_percentile, score_edit, score_all,
    pending_metrics, summarise,
)

__all__ = [
    "natural_seam_distribution", "seam_percentile", "score_edit", "score_all",
    "pending_metrics", "summarise",
]
