import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner',SCRIPTS/'run_progressive_10_datasets.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_default_profiles_have_ten_datasets(self):
        profiles=json.loads((SCRIPTS/'progressive_10_profiles.json').read_text())
        runner.validate_profiles(profiles)
        self.assertEqual(len(profiles),10)
        self.assertNotIn('nuswide',profiles);self.assertNotIn('spacev1b',profiles)
        self.assertTrue(all(p['k']==1 for p in profiles.values()))
        self.assertFalse(profiles['sift1b']['ppd'])

    def test_invalid_pivot_configuration_rejected(self):
        profiles=json.loads((SCRIPTS/'progressive_10_profiles.json').read_text())
        profiles['sift1m']['pmax']=1
        with self.assertRaises(ValueError):runner.validate_profiles(profiles)

    def test_dry_run_has_exploration_and_ablation_flags(self):
        text=subprocess.check_output([sys.executable,str(SCRIPTS/'run_progressive_10_datasets.py'),
            '--data-root','/unused/data','--log-dir','/unused/results','--datasets','sift1m',
            '--blocks','8','--greedy-ablation','--dry-run'],text=True)
        frozen=[line for line in text.splitlines() if '--projection_progressive frozen' in line]
        self.assertTrue(any('--projection_progressive_keep_exploration' in line for line in frozen))
        self.assertTrue(any('--projection_progressive_keep_exploration' not in line for line in frozen))

    def test_groundtruth_rows_follow_query_split(self):
        for extension in ['bin','ivecs','i32bin']:
            with self.subTest(extension=extension),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);data=root/'data'/'glove25'
                (data/'origin').mkdir(parents=True);(data/'result').mkdir()
                vectors=b''.join(struct.pack('<Iff',2,float(qi),float(qi+1)) for qi in range(6))
                (data/'origin/glove25_base.fvecs').write_bytes(vectors)
                (data/'origin/glove25_query.fvecs').write_bytes(vectors)
                gt=data/'result'/f'groundtruth_1.{extension}'
                if extension=='bin':gt.write_bytes(struct.pack('<6q',*range(10,16))+struct.pack('<6f',*range(6)))
                elif extension=='ivecs':gt.write_bytes(b''.join(struct.pack('<Iii',2,10+qi,100+qi) for qi in range(6)))
                else:gt.write_bytes(struct.pack('<II',6,2)+b''.join(struct.pack('<ii',10+qi,100+qi) for qi in range(6)))
                profile={'glove25':{'normalize':False,'input_format':'fvecs','k':1,
                                   'groundtruth':'' if extension=='bin' else f'result/{gt.name}'}}
                config=root/'config.json';config.write_text(json.dumps(profile))
                subprocess.run([sys.executable,str(SCRIPTS/'prepare_progressive_query_split.py'),
                    '--data-root',str(root/'data'),'--output-root',str(root/'views'),
                    '--config',str(config),'--fraction','.33'],check=True,stdout=subprocess.DEVNULL)
                selected_sets=[]
                for split in ['tune','eval']:
                    output=root/'views'/split/'glove25'
                    selected=json.loads((output/'query_ids.json').read_text())['query_ids'];selected_sets.append(set(selected))
                    self.assertEqual((output/'origin/glove25_query.fvecs').read_bytes(),
                                     b''.join(vectors[qi*12:(qi+1)*12] for qi in selected))
                    raw=(output/'result'/gt.name).read_bytes()
                    if extension=='bin':
                        ids=struct.unpack('<'+'q'*len(selected),raw[:8*len(selected)])
                        self.assertEqual(list(ids),[10+qi for qi in selected])
                        distances=struct.unpack('<'+'f'*len(selected),raw[8*len(selected):])
                        self.assertEqual(list(distances),list(map(float,selected)))
                    elif extension=='ivecs':
                        self.assertEqual(raw,b''.join(struct.pack('<Iii',2,10+qi,100+qi) for qi in selected))
                    else:
                        self.assertEqual(raw,struct.pack('<II',len(selected),2)+b''.join(struct.pack('<ii',10+qi,100+qi) for qi in selected))
                self.assertFalse(selected_sets[0]&selected_sets[1])
                self.assertEqual(selected_sets[0]|selected_sets[1],set(range(6)))


if __name__=='__main__':unittest.main()
