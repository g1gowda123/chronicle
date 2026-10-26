import sys
from projectaria_tools.utils.vrs_to_mp4_utils import convert_vrs_to_mp4

src = sys.argv[1] if len(sys.argv) > 1 else "/data/aria_aea_test.vrs"
dst = sys.argv[2] if len(sys.argv) > 2 else "/data/aria_aea_test.mp4"
out = sys.argv[3] if len(sys.argv) > 3 else "/data"
convert_vrs_to_mp4(src, dst, out, 1)
print("done", dst)
