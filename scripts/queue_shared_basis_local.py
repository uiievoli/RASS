#!/usr/bin/env python3
"""Yield an existing local queue at a child-process boundary; always resume it."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def state(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
    except FileNotFoundError:return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yield-pid',type=int,required=True)
    parser.add_argument('--log-dir',type=Path,required=True)
    args=parser.parse_args();stopped=[]
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        rows=subprocess.check_output(['ps','-eo','pid,ppid,pgid,comm'],text=True).splitlines()[1:]
        members=[(int(fields[0]),fields[3]) for line in rows if len(fields:=line.split())==4 and int(fields[2])==args.yield_pid]
        # Pause only waiting orchestration parents. Timed query children finish
        # uninterrupted; their wall-clock measurements are never stretched.
        for pid,comm in members:
            if comm in ['bash','python3']:
                os.kill(pid,signal.SIGSTOP);stopped.append(pid)
        children=[pid for pid,comm in members if comm in ['query','ppd_query']]
        print('Paused orchestration parents:',stopped,'waiting for current children:',children,flush=True)
        last=0
        while any(state(pid) not in [None,'Z'] for pid in children):
            if time.time()-last>60:
                print(time.strftime('[%H:%M:%S]'),'Current query batch still running; comparison queued',flush=True);last=time.time()
            time.sleep(15)
        print('Current batch finished; CPU slot available',flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/test_shared_basis_smoke.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_shared_basis_global.py'),
                        '--log-dir',str(args.log_dir)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_shared_basis_global.py'),str(args.log_dir)],cwd=ROOT,check=True)
        (args.log_dir/'queue.exitcode').write_text('0\n')
    finally:
        for pid in reversed(stopped):
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        print('Resumed previous orchestration parents:',stopped,flush=True)


if __name__=='__main__':main()
