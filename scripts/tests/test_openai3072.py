import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import hashlib
import json
import numpy as np
import h5py

SCRIPTS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('openai_fine',SCRIPTS/'run_openai3072_fine_pivot.py')
fine=importlib.util.module_from_spec(spec);spec.loader.exec_module(fine)


class OpenAI3072Tests(unittest.TestCase):
    def test_grid_and_refinement_cover_limits(self):
        grid=fine.pivot_grid(512,4)
        self.assertEqual(len(grid),129)
        self.assertEqual(grid[:5],[2,4,8,12,16])
        self.assertEqual(grid[-1],512)
        self.assertEqual(fine.pivot_grid(512,1),list(range(2,513)))
        refined=fine.refinement_grid([2,512],512,2)
        self.assertEqual(refined,[2,3,4,510,511,512])

    def test_bad_grid_rejected(self):
        for maximum,step in [(513,4),(1,1),(256,0)]:
            with self.assertRaises(ValueError):fine.pivot_grid(maximum,step)

    def test_normalized_hdf_export_preserves_official_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'fixture.hdf5'
            train=np.eye(4,dtype=np.float32);queries=train[:2]
            neighbors=np.array([[0,1],[1,0]],dtype=np.int64)
            with h5py.File(source,'w') as f:
                f.attrs['distance']='normalized';f['train']=train;f['test']=queries;f['neighbors']=neighbors
            checksum=hashlib.sha256(source.read_bytes()).hexdigest()
            command=[sys.executable,str(SCRIPTS/'prepare_cosine_hdf5.py'),'--hdf5',str(source),'--dataset','unit-test',
                     '--dimension','4','--data-root',str(root/'data'),'--sha256',checksum]
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            data=root/'data/unit-test'
            raw=np.fromfile(data/'origin/unit-test_base.fvecs',dtype='<f4').reshape(4,5)
            np.testing.assert_array_equal(raw[:,1:],train)
            gt=np.fromfile(data/'result/official_neighbors.i32bin',dtype='<i4')
            np.testing.assert_array_equal(gt[:2],[2,2]);np.testing.assert_array_equal(gt[2:].reshape(2,2),neighbors)
            # An existing export marker must not bypass a changed expected hash.
            failed=subprocess.run(command[:-1]+['0'*64],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            self.assertNotEqual(failed.returncode,0)

    def test_false_unit_norm_declaration_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'invalid.hdf5'
            with h5py.File(source,'w') as f:
                f.attrs['distance']='normalized';f['train']=np.ones((4,4),dtype=np.float32)
                f['test']=np.ones((2,4),dtype=np.float32);f['neighbors']=np.array([[0],[1]])
            result=subprocess.run([sys.executable,str(SCRIPTS/'prepare_cosine_hdf5.py'),'--hdf5',str(source),
                '--dataset','bad','--dimension','4','--data-root',str(root/'data')],
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('non-unit vectors',result.stderr)


if __name__=='__main__':unittest.main()
