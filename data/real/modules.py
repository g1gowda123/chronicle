import pkgutil

import projectaria_tools

print(projectaria_tools.__file__)
names = [m.name for m in pkgutil.walk_packages(projectaria_tools.__path__, "projectaria_tools.")]
for name in names:
    if "mp4" in name or "vrs" in name.lower():
        print(name)
print("--- utils ---")
for name in names:
    if ".utils." in name and name.count(".") <= 3:
        print(name)
