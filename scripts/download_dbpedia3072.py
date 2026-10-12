#!/usr/bin/env python3
"""Resumable, SHA256-verified download of DBpedia OpenAI-3072 (63 shards)."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--raw-dir',type=Path)
    parser.add_argument('--endpoint',default='https://huggingface.co')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    manifest_path=here/'dbpedia3072_download_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    raw=args.raw_dir or args.data_root/'.downloads/dbpedia3072'
    if not args.dry_run:
        raw.mkdir(parents=True,exist_ok=True)
        lock=(raw/'.download.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for number,entry in enumerate(manifest['files'],1):
        path=raw/Path(entry['path']).name
        partial=path.with_suffix('.parquet.partial')
        url=f'{args.endpoint.rstrip("/")}/datasets/{manifest["repo"]}/resolve/{manifest["revision"]}/{entry["path"]}?download=true'
        command=['curl','--fail','--location','--retry','8','--retry-all-errors',
                 '--connect-timeout','30','--continue-at','-','--output',str(partial),url]
        print(f'SHARD {number}/{len(manifest["files"])} {path.name}',flush=True)
        if args.dry_run:print(shlex.join(command));continue
        if path.exists():
            if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
                raise ValueError(f'Corrupt completed shard (preserved): {path}')
            print('SKIP verified',path.name,flush=True);continue
        complete=partial.exists() and partial.stat().st_size==entry['bytes']
        if not complete:subprocess.run(command,check=True)
        if partial.stat().st_size!=entry['bytes'] or digest(partial)!=entry['sha256']:
            raise ValueError(f'Checksum/size mismatch (preserved): {partial}')
        partial.replace(path)
    command=[sys.executable,str(here/'prepare_dbpedia3072.py'),'--parquet-dir',str(raw),
             '--data-root',str(args.data_root),'--manifest',str(manifest_path)]
    if args.dry_run:print(shlex.join(command))
    else:subprocess.run(command,check=True)


if __name__=='__main__':main()
