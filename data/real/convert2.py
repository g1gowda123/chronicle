import sys

from moviepy import VideoClip
from projectaria_tools.tools.vrs_to_mp4.vrs_to_mp4_utils import Vrs2Mp4Converter

src = sys.argv[1] if len(sys.argv) > 1 else "/data/aria_aea_test.vrs"
dst = sys.argv[2] if len(sys.argv) > 2 else "/data/aria_aea_test.mp4"

converter = Vrs2Mp4Converter(src, "214-1", 1, [0, 2])
duration = converter.duration_ns() * 1e-9
print("duration", duration, "fps", converter.video_fps())
clip = VideoClip(converter.make_frame, duration=duration)
clip.write_videofile(dst, fps=converter.video_fps(), codec="libx264", audio=False, logger="bar")
clip.close()
print("done", dst)
