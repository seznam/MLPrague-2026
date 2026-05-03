import os
import re
import zipfile
import io


def _get_version():
    pat = r"""
        (?P<version>\d+\.\d+)         # minimum 'N.N'
        (?P<extraversion>(?:\.\d+)*)  # any number of extra '.N' segments
        (?:
            (?P<prerel>[abc]|rc)      # 'a' = alpha, 'b' = beta
                                      # 'c' or 'rc' = release candidate
            (?P<prerelversion>\d+(?:\.\d+)*)
        )?
        (?P<postdev>(\.post(?P<post>\d+))?(\.dev(?P<dev>\d+))?)?
    """

    def search_in_file(fd):
        for line in fd:
            if not line.startswith('##'):
                continue
            match = re.search(pat, line, re.VERBOSE)
            if match:
                return match.group()
        raise ValueError("Can't get version")

    package_root_folder = os.path.dirname(os.path.dirname(__file__))
    is_zipped = zipfile.is_zipfile(package_root_folder)
    if is_zipped:
        filename = os.path.join(os.path.basename(os.path.dirname(__file__)), 'CHANGELOG.md')
        with zipfile.ZipFile(package_root_folder) as zipped_folder:
            with io.TextIOWrapper(zipped_folder.open(filename, 'r'), encoding='utf-8') as file:
                return search_in_file(file)
    else:
        filename = os.path.join(os.path.dirname(__file__), 'CHANGELOG.md')
        with open(filename, 'r') as file:
            return search_in_file(file)


__version__ = _get_version()
