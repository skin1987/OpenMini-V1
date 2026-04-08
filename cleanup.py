#!/ 清理重复目录
# 删除 hardware/memory, hardware/thread, service 目录

import os
import shutil

dirs_to_remove = [
    "openmini-server/src/hardware/memory",
    "openmini-server/src/hardware/thread",
    "openmini-server/src/service"
]

for dir in dirs_to_remove:
    path = os.path.join(os.getcwd(), dir)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed: {path}")
    else:
        print(f"Not found: {path}")

print("Cleanup complete!")
