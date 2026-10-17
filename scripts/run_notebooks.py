"""Выполнить ноутбуки последовательно и сохранить результаты, включая вывод при ошибке."""
from pathlib import Path
import argparse
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = [
    'research/01_data_overview.ipynb',
    'features/01_feature_engineering.ipynb',
    'training/01_linear.ipynb',
    'training/02_catboost.ipynb',
    'training/03_dcn.ipynb',
    'training/04_comparison.ipynb',
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', help='Пути от корня проекта; по умолчанию — весь pipeline')
    parser.add_argument('--kernel', default='python3')
    args = parser.parse_args()
    for relative in args.notebooks or DEFAULT:
        path = ROOT / relative
        notebook = nbformat.read(path, as_version=4)
        started = time.perf_counter()
        print(f'START {relative}', flush=True)
        try:
            NotebookClient(notebook, timeout=1800, kernel_name=args.kernel,
                           resources={'metadata': {'path': str(ROOT)}}).execute()
        finally:
            nbformat.write(notebook, path)
        print(f'DONE {relative}: {time.perf_counter() - started:.1f}s', flush=True)

if __name__ == '__main__':
    main()
