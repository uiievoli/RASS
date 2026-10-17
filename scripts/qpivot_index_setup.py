"""Find compatible persisted PCA banks, or build them through native query.cpp."""
from pathlib import Path
import struct


def index_header(path):
    try:
        with path.open('rb') as stream:
            raw=stream.read(160)
        if raw[:8]!=b'TRIBASE2' or struct.unpack_from('<I',raw,8)[0]!=10:return None
        d,nlist=struct.unpack_from('<QQ',raw,16)
        metric,opt=struct.unpack_from('<II',raw,32)
        scope=struct.unpack_from('<I',raw,64)[0]
        count=struct.unpack_from('<Q',raw,68)[0]
        length=struct.unpack_from('<I',raw,84)[0]
        if length>64 or len(raw)<88+length:return None
        return dict(d=d,nlist=nlist,metric=metric,opt=opt,scope=scope,pmax=count,
                    method=raw[88:88+length].decode())
    except (OSError,ValueError,struct.error):return None


def prepare_global_index(dataset_dir,nlist,pmax,normalize,perf_dir,log_dir,execute,
                         explicit=None,index_roots=(),require_groundtruth=False,
                         groundtruth=None,rebuild=False):
    """execute(command, log) runs already configured native executables.

    With rebuild=True, build only from raw fvecs into log_dir/indexes.
    Legacy callers otherwise reuse rich bank -> upgrade Triangle -> build fvecs.
    Default full-query GT is also generated if requested and absent.
    """
    dataset_dir=Path(dataset_dir);dataset=dataset_dir.name
    if rebuild and (explicit or index_roots):
        raise ValueError('fresh Global build forbids --index/--index-root reuse; use a new log directory')
    tag='_unit' if normalize else ''
    query=Path(perf_dir)/'bin/query'
    upgrade=Path(perf_dir)/'bin/build_global_pca_index'
    directories=[dataset_dir/'index',dataset_dir.parent/'rass_index_cache'/dataset]
    for root in index_roots:
        root=Path(root)
        directories.extend([root/dataset,root/dataset/'index',root])
    paths=[] if rebuild else sorted(set(path for directory in directories for path in directory.glob('*.index')))
    compatible=[]
    for path in paths:
        h=index_header(path)
        if h and h['nlist']==nlist and h['metric']==1 and h['opt']&1 and ('unit' in path.name)==normalize:
            compatible.append((path,h))
    if explicit:
        index=Path(explicit).resolve();h=index_header(index)
        if not h or h['nlist']!=nlist or h['metric']!=1 or not h['opt']&1 or h['scope']!=0 or h['method']!='pca' or h['pmax']!=pmax:
            raise ValueError(f'explicit index must be a compatible Global PCA P{pmax} bank: {index}')
    else:
        rich=[path for path,h in compatible if h['scope']==0 and h['method']=='pca' and h['pmax']==pmax]
        index=rich[0] if rich else None

    native=[query,'--benchmarks_path',dataset_dir.parent,'--dataset',dataset,
            '--input_format','fvecs','--output_format','bin','--metric','l2','--k','1',
            '--nq','0','--nlist',nlist,'--nprobes','1','--cache','--train_only',
            '--opt_levels','OPT_TRIANGLE','--multipivot_modes','projection',
            '--multipivot_scope','global','--multipivot_method','pca','--pivot_counts',pmax,
            '--pivot_seed','0','--signature_precision','float32','--verbose']
    if normalize:native+=['--normalize']
    if rebuild:
        # Isolate this run from all dataset/index and historical cache files.
        # No --cache, --from_index, --load_index or centroid file on this path.
        base=dataset_dir/'origin'/f'{dataset}_base.fvecs'
        if not base.is_file():raise FileNotFoundError(f'fresh build requires base vectors: {base}')
        directory=Path(log_dir)/'indexes'/dataset
        directory.mkdir(parents=True,exist_ok=True)
        index=directory/f'global_pca_nlist{nlist}_P{pmax}_seed0{tag}.index'
        if index.exists() or Path(str(index)+'.partial').exists():
            raise ValueError(f'fresh index output already exists; use a new log directory: {index}')
        print(f'FRESH BUILD Global PCA Pmax={pmax}, directions={pmax-1}: {base}',flush=True)
        execute([*[arg for arg in native if arg!='--cache'],'--index_path',index],
                Path(log_dir)/f'build_index_{dataset}.log')
        h=index_header(index)
        if not h or (h['nlist'],h['scope'],h['method'],h['pmax'],h['metric'])!=(nlist,0,'pca',pmax,1) or not h['opt']&1:
            raise RuntimeError(f'fresh builder produced incompatible Global PCA index: {index}')
    if index is None:
        directory=dataset_dir/'index';directory.mkdir(parents=True,exist_ok=True)
        index=directory/f'qpivot_v10_nlist_{nlist}{tag}_mp_global_pca_P{pmax}_seed0.index'
        if compatible:
            source=min(compatible,key=lambda item:(item[1]['pmax']!=0,str(item[0])))[0]
            print(f'UPGRADE existing Triangle index: {source}',flush=True)
            execute([upgrade,source,index,pmax],log_dir/f'build_index_{dataset}.log')
        else:
            base=dataset_dir/'origin'/f'{dataset}_base.fvecs'
            if not base.is_file():raise FileNotFoundError(f'no reusable index and missing base vectors: {base}')
            if not query.is_file():raise FileNotFoundError(f'missing native builder {query}; rerun without --skip-build')
            print(f'BUILD missing Global PCA index from base vectors: {base}',flush=True)
            execute([*native,'--index_path',index],log_dir/f'build_index_{dataset}.log')
    print(f'{"BUILT fresh" if rebuild else "REUSE"} Global PCA index: {index}',flush=True)
    if require_groundtruth:
        gt=Path(groundtruth) if groundtruth else dataset_dir/'result'/f'groundtruth_1{tag}.bin'
        if not gt.is_file():
            if groundtruth:raise FileNotFoundError(f'missing explicitly supplied groundtruth: {gt}')
            if not query.is_file():raise FileNotFoundError(f'missing native GT builder {query}; rerun without --skip-build')
            print(f'BUILD missing full-query groundtruth: {gt}',flush=True)
            execute([*native,'--load_index',index],log_dir/f'build_groundtruth_{dataset}.log')
            if not gt.is_file():raise RuntimeError(f'native query did not create groundtruth: {gt}')
    return index.resolve()
