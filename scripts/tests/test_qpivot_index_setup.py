import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qpivot_index_setup',SCRIPTS/'qpivot_index_setup.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)


def write_index(path,p=5,scope=0,nlist=4):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=bytearray(160);raw[:8]=b'TRIBASE2'
    struct.pack_into('<IIQQII',raw,8,10,0x01020304,8,nlist,1,1)
    struct.pack_into('<IQQI',raw,64,scope,p,0,3);raw[88:91]=b'pca'
    path.write_bytes(raw)


class AutomaticIndexSetup(unittest.TestCase):
    def test_forced_build_ignores_old_rich_and_triangle_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ds=root/'data/demo';origin=ds/'origin'
            origin.mkdir(parents=True);(origin/'demo_base.fvecs').touch()
            old=ds/'index/global_P5.index';write_index(old)
            old_bytes=old.read_bytes();logs=root/'logs';logs.mkdir();commands=[]
            def execute(cmd,log):
                cmd=list(map(str,cmd));commands.append(cmd)
                for forbidden in ['--cache','--load_index','--from_index','--centroids_path']:
                    self.assertNotIn(forbidden,cmd)
                write_index(Path(cmd[cmd.index('--index_path')+1]))
            index=setup.prepare_global_index(ds,4,5,False,root/'build',logs,execute,rebuild=True)
            self.assertEqual(index.parent,logs/'indexes/demo')
            self.assertEqual(len(commands),1);self.assertEqual(old.read_bytes(),old_bytes)
            with self.assertRaisesRegex(ValueError,'already exists'):
                setup.prepare_global_index(ds,4,5,False,root/'build',logs,execute,rebuild=True)

    def test_forced_build_rejects_reuse_arguments(self):
        with self.assertRaisesRegex(ValueError,'forbids'):
            setup.prepare_global_index(Path('/tmp/demo'),4,5,False,Path('/tmp'),Path('/tmp'),
                lambda *args:None,explicit=Path('/tmp/old.index'),rebuild=True)

    def test_fresh_normalized_data_builds_index_and_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ds=root/'data/dbpedia3072m_holdout';origin=ds/'origin'
            origin.mkdir(parents=True);(origin/f'{ds.name}_base.fvecs').touch()
            perf=root/'build/perf';(perf/'bin').mkdir(parents=True);(perf/'bin/query').touch()
            logs=root/'logs';logs.mkdir();commands=[]
            def execute(cmd,log):
                cmd=list(map(str,cmd));commands.append(cmd)
                self.assertIn('--normalize',cmd);self.assertIn('--train_only',cmd)
                if '--index_path' in cmd:write_index(Path(cmd[cmd.index('--index_path')+1]))
                gt=ds/'result/groundtruth_1_unit.bin';gt.parent.mkdir(exist_ok=True);gt.touch()
            index=setup.prepare_global_index(ds,4,5,True,perf,logs,execute,require_groundtruth=True)
            self.assertTrue(index.exists());self.assertEqual(len(commands),1)
            self.assertIn('--nq',commands[0]);self.assertEqual(commands[0][commands[0].index('--nq')+1],'0')
            commands.clear()
            self.assertEqual(setup.prepare_global_index(ds,4,5,True,perf,logs,execute,require_groundtruth=True),index)
            self.assertFalse(commands)

    def test_cache_rich_bank_is_reused_without_p0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ds=root/'data/demo';bank=root/'data/rass_index_cache/demo/native_unit_P5.index'
            write_index(bank);commands=[]
            result=setup.prepare_global_index(ds,4,5,True,root/'build',root,
                lambda *args:commands.append(args))
            self.assertEqual(result,bank.resolve());self.assertFalse(commands)

    def test_any_compatible_triangle_can_be_upgraded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ds=root/'data/demo';bank=ds/'index/per_list_pca_P9.index'
            write_index(bank,p=9,scope=1);commands=[]
            def execute(cmd,log):
                commands.append(cmd);write_index(Path(cmd[2]))
            result=setup.prepare_global_index(ds,4,5,False,root/'build',root,execute)
            self.assertTrue(result.exists());self.assertEqual(len(commands),1)
            self.assertEqual(Path(commands[0][0]).name,'build_global_pca_index')
            self.assertEqual(commands[0][1],bank)


if __name__=='__main__':unittest.main()
