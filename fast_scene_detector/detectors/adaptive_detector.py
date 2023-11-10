import torch

from .content_detector import ContentDetector


class AdaptiveDetector(ContentDetector):
    def __init__(
        self,
        adaptive_threshold: float = 3.0,
        min_scene_len: int = 15,
        window_width: int = 2,
        min_content_val: float = 15.0,
    ):
        if window_width < 1:
            raise ValueError('window_width must be at least 1.')

        super().__init__(
            threshold=255.0,
            min_scene_len=0,
        )

        self.adaptive_threshold = adaptive_threshold
        self.min_scene_len = min_scene_len
        self.window_width = window_width
        self.min_content_val = min_content_val

        self._buffer: list[tuple[int, float]] = []
        self._last_cut: int | None = None

    def process_frame(self, frame_idx: int, frame: torch.Tensor) -> tuple[int | None, float]:
        _, score = super().process_frame(frame_idx, frame)

        required_frames = 1 + (2 * self.window_width)
        self._buffer.append((frame_idx, score))

        if len(self._buffer) < required_frames:
            return None, 0.0

        # Remove all frames from the buffer that are outside the window.
        self._buffer = self._buffer[-required_frames:]

        average_score = sum(
            item[1] for i, item in enumerate(self._buffer) if i != self.window_width
        ) / (2 * self.window_width)
        average_score_is_zero = abs(average_score) < 0.00001

        target = self._buffer[self.window_width]
        if not average_score_is_zero:
            adaptive_ratio = min(target[1] / average_score, 255.0)
        elif average_score_is_zero and target[1] >= self.min_content_val:
            # if we would have divided by zero, set adaptive_ratio to the max (255.0)
            adaptive_ratio = 255.0
        else:
            adaptive_ratio = 0.0

        if adaptive_ratio >= self.adaptive_threshold and target[1] >= self.min_content_val:
            if self._last_cut is None or (target[0] - self._last_cut) >= self.min_scene_len:
                self._last_cut = target[0]
                return target[0], target[1]

        return None, 0.0
