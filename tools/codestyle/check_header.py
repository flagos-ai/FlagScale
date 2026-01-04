import datetime
import os
import re
import sys

COPYRIGHT = """Copyright © {year} BAAI. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License")"""


RE_FS_HEADER = re.compile(r".*Copyright \(c\) \d{4}", flags=re.IGNORECASE)
RE_ENCODE = re.compile(r"^[\t\v ]*#.*?coding[:=]", flags=re.IGNORECASE)
RE_SHEBANG = re.compile(r"^[\t\v ]*#[ \t]?\!")


def _check_header(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[:5]:
            if not lines:
                return False
            if RE_FS_HEADER.search(line) is not None:
                return True
    return False


def generate_header(path):
    with open(path, "r+", encoding="utf-8") as f:
        lines = f.readlines()

        insert_index = 0
        for i, line in enumerate(lines[:5]):
            if RE_ENCODE.search(line) or RE_SHEBANG.search(line):
                insert_index = i + 1

        f.seek(0, 0)
        year = datetime.datetime.now().year

        copyright = ""
        for l in COPYRIGHT.format(year=year).splitlines():
            if l:
                copyright += f"# {l}{os.linesep}"
            else:
                copyright += f"#{os.linesep}"

        if insert_index == 0:
            f.write(copyright + os.linesep)
            f.writelines(lines)
        else:
            heads = lines[:insert_index]
            f.write(heads)
            f.write(copyright + os.linesep)
            f.writelines(lines[insert_index:])


def main():
    for path in sys.argv[1:]:
        if _check_header(path):
            continue
        generate_header(path)


if __name__ == "__main__":
    sys.exit(main())
