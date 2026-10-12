import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class DBpediaExportTests(unittest.TestCase):
    def test_disjoint_export_and_reuse(self):
        script=Path(__file__).resolve().parents[1]/'prepare_dbpedia3072.py'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            column='text-embedding-3-large-3072-embedding'
            vectors=[[float(i+1),2.0,3.0] for i in range(6)]
            pq.write_table(pa.table({column:vectors}),root/'train.parquet')
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(dimension=3,rows=6,embedding_column=column,
                files=[dict(path='data/train.parquet')],revision='fixture')))
            command=[sys.executable,str(script),'--parquet-dir',str(root),'--data-root',str(root/'data'),
                     '--manifest',str(manifest),'--nq','2']
            subprocess.run(command,check=True,capture_output=True)
            origin=root/'data/dbpedia3072m_holdout/origin'
            def rows(name):
                values=np.frombuffer((origin/f'dbpedia3072m_holdout_{name}.fvecs').read_bytes(),
                     dtype=np.dtype([('d','<i4'),('v','<f4',(3,))]))
                self.assertTrue(np.all(values['d']==3))
                return values['v'][:,0].tolist()
            base,query=rows('base'),rows('query')
            self.assertEqual(len(base),4);self.assertEqual(len(query),2)
            self.assertFalse(set(base)&set(query))
            self.assertEqual(sorted(base+query),[1.,2.,3.,4.,5.,6.])
            result=subprocess.run(command,check=True,capture_output=True,text=True)
            self.assertIn('REUSE',result.stdout)


if __name__=='__main__':unittest.main()
