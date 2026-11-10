"""Restore ELF aliases flattened into separate files by the OctoMap wheel.

HPP-FCL requests liboctomap.so while CMake requests its versioned SONAME.
Separate byte-identical files can then be loaded twice and register the same
static string destructor twice. Symlinks let the loader identify one inode.
Only identical files inside this project's virtual environment are replaced.
"""
import hashlib
import json
from pathlib import Path
import sys


def repair():
    root = Path(__file__).resolve().parents[1]
    prefix = root/'runtime/venv'
    if Path(sys.prefix).resolve() != prefix.resolve():
        raise RuntimeError('Run with this project\'s runtime/venv Python')
    library = prefix/'lib/python3.8/site-packages/cmeel.prefix/lib'
    repaired = []
    for name in ['liboctomap', 'liboctomath']:
        paths = sorted(library.glob(name+'.so*'), key=lambda p: len(p.name), reverse=True)
        if not paths:
            raise RuntimeError('Missing OctoMap library: '+name)
        canonical = paths[0]
        if canonical.is_symlink():
            canonical = canonical.resolve()
        digest = hashlib.sha256(canonical.read_bytes()).digest()
        for alias in paths:
            if alias == canonical:
                continue
            if alias.is_symlink() and alias.resolve() == canonical.resolve():
                continue
            if hashlib.sha256(alias.read_bytes()).digest() != digest:
                raise RuntimeError('Refusing to replace non-identical OctoMap library: '+str(alias))
            alias.unlink()
            alias.symlink_to(canonical.name)
            repaired.append(str(alias.relative_to(prefix)))
    return {'repaired_aliases':repaired, 'canonical_libraries_unchanged':True}


if __name__ == '__main__':
    print(json.dumps(repair(), indent=2))
