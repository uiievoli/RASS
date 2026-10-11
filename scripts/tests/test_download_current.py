import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('download_current', Path(__file__).resolve().parents[1] / 'download_current_datasets.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DownloadReadinessTests(unittest.TestCase):
    def write_vectors(self, path, rows, dimension):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((struct.pack('<i', dimension) + bytes(4 * dimension)) * rows)

    def test_suite(self):
        self.assertEqual(len(download.DATASETS), 11)
        self.assertEqual(len(download.DEFAULT_DATASETS), 10)
        self.assertNotIn('sift1b', download.DEFAULT_DATASETS)
        self.assertIn('sift1b', download.DATASETS)
        self.assertNotIn('nuswide', download.DATASETS)
        self.assertNotIn('spacev1b', download.DATASETS)
        self.assertIn('HandOutlines', download.DATASETS)

    def test_native_complete_and_truncated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin = root / 'sift1m/origin'
            self.write_vectors(origin / 'sift1m_base.fvecs', 3, 128)
            query = origin / 'sift1m_query.fvecs'
            self.write_vectors(query, 2, 128)
            self.assertTrue(download.prepared(root, 'sift1m'))
            query.write_bytes(query.read_bytes()[:-1])
            self.assertFalse(download.prepared(root, 'sift1m'))

    def test_cosine_requires_export_marker_and_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = 'simplewiki-openai-3072-normalized'
            origin = root / dataset / 'origin'
            self.write_vectors(origin / f'{dataset}_base.fvecs', 3, 3072)
            self.write_vectors(origin / f'{dataset}_query.fvecs', 2, 3072)
            self.assertFalse(download.prepared(root, dataset))
            metadata = dict(base_vectors=3, queries=2, neighbor_width=1,
                            sha256=download.CHECKSUMS[dataset])
            (origin / 'cosine_source.json').write_text(json.dumps(metadata))
            truth = root / dataset / 'result/official_neighbors.i32bin'
            truth.parent.mkdir()
            truth.write_bytes(struct.pack('<IIII', 2, 1, 0, 1))
            self.assertTrue(download.prepared(root, dataset))
            metadata['sha256'] = 'wrong'
            (origin / 'cosine_source.json').write_text(json.dumps(metadata))
            self.assertFalse(download.prepared(root, dataset))


if __name__ == '__main__':
    unittest.main()
