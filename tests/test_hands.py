from __future__ import annotations

import cv2
import numpy as np

from chronicle.perception.config import HandConfig
from chronicle.perception.hands import ColorHands
from chronicle.perception.palette import PALETTE
from chronicle.types import BBox


def test_color_hand_landmarks_span_the_skin_hull() -> None:
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    cv2.ellipse(frame, (160, 130), (30, 40), 0, 0, 360, PALETTE["hand"], -1)
    for offset in (-20, -8, 8, 20):
        cv2.circle(frame, (160 + offset, 82), 10, PALETTE["hand"], -1)

    hands = ColorHands(HandConfig()).detect(frame, frame_index=0)

    assert len(hands) == 1
    hand = hands[0]
    assert len(hand.landmarks) == 21
    # Regression: pseudo-landmarks used to collapse onto the wrist-to-tip axis,
    # reporting a bbox far narrower than the skin region and suppressing IoS.
    assert hand.bbox.ios(BBox(130, 72, 190, 170)) > 0.9
    assert hand.index_tip[1] < hand.wrist[1]  # image coords: tip is above wrist
    assert hand.source == "color-skin"
