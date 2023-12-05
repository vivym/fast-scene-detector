from tqdm import tqdm

from .decoders import VPFDecoder
from .detectors import AdaptiveDetector


def detect(video_path: str, gpu_id: int = 0, verbose: bool = False) -> list[tuple[int, int]]:
    decoder = VPFDecoder(video_path, gpu_id=gpu_id)
    detector = AdaptiveDetector()

    scenes = []
    last_cut_idx = 0

    for frame_idx, frame in tqdm(enumerate(decoder.iter_frames(pixel_format="hsv")), disable=not verbose):
        cut_idx, _ = detector.process_frame(frame_idx, frame)
        if cut_idx is not None:
            scenes.append((last_cut_idx, cut_idx))
            last_cut_idx = cut_idx

    if frame_idx != last_cut_idx:
        scenes.append((last_cut_idx, frame_idx))

    return scenes
