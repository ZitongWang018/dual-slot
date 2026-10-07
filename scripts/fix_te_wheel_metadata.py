"""Correct TE 2.13 cu12's internal tag to match its py3-none wheel filename.

The cu12 package contains a Python-independent C++ shared library. Python
bindings are a separate, correctly tagged transformer_engine_torch wheel.
Recent pip check detects the upstream core wheel's stale cp310 metadata.
Only that exact inconsistency is repaired; no binary is changed.
"""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import subprocess


def main():
    package = metadata.distribution('transformer_engine_cu12')
    if package.version != '2.13.0':
        return
    wheel_relative = next(f for f in package.files if str(f).endswith('.dist-info/WHEEL'))
    wheel_path = package.locate_file(wheel_relative)
    text = wheel_path.read_text()
    old = 'Tag: cp310-cp310-manylinux_2_28_x86_64'
    if old not in text:
        return
    libraries = [package.locate_file(f) for f in package.files if str(f).endswith('.so')]
    if len(libraries) != 1 or libraries[0].name != 'libtransformer_engine.so':
        raise RuntimeError('Unexpected TE core binary layout; refusing tag correction')
    library = libraries[0]
    dependencies = subprocess.check_output(['readelf', '-d', str(library)], text=True)
    symbols = subprocess.check_output(['nm', '-D', str(library)], text=True)
    if 'libpython' in dependencies or 'PyInit_' in symbols:
        raise RuntimeError('Core library has Python ABI dependencies; refusing correction')
    binary_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    updated = text.replace(old, 'Tag: py3-none-manylinux_2_28_x86_64').encode()
    wheel_path.with_name('WHEEL.original').write_text(text)
    wheel_path.write_bytes(updated)
    record_path = wheel_path.with_name('RECORD')
    rows = list(csv.reader(io.StringIO(record_path.read_text())))
    for row in rows:
        if row[0] == str(wheel_relative):
            row[1] = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(updated).digest()).decode().rstrip('=')
            row[2] = str(len(updated))
    output = io.StringIO(); csv.writer(output).writerows(rows)
    record_path.write_text(output.getvalue())
    assert hashlib.sha256(library.read_bytes()).hexdigest() == binary_hash
    print(json.dumps({'te_core_wheel_metadata_corrected': True, 'binary_unchanged_sha256': binary_hash,
                      'from': old, 'to': 'Tag: py3-none-manylinux_2_28_x86_64'}), flush=True)


if __name__ == '__main__':
    main()
