"""Keep Apex's custom-op annotations compatible with pinned PyTorch 2.6.

typing.List[int] and list[int] describe the same type, but PyTorch 2.6's
schema parser recognizes only typing.List[int]. No operator code is changed.
"""
import importlib.metadata


def main():
    package = importlib.metadata.distribution('apex')
    path = package.locate_file('apex/normalization/fused_layer_norm.py')
    source = path.read_text()
    if 'list[int]' not in source:
        return
    if 'from typing import List, Tuple' not in source:
        raise RuntimeError('Unexpected Apex source; refusing annotation correction')
    path.with_suffix('.py.original').write_text(source)
    count = source.count('list[int]')
    path.write_text(source.replace('list[int]', 'List[int]'))
    print(f'Apex PyTorch 2.6 annotation compatibility: {count} annotations corrected')


if __name__ == '__main__':
    main()
