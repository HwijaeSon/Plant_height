"""Wait for frozen-choice evaluation, then audit, export and optionally build the paper."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from run_observation_benchmark import ROOT, OUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tectonic',type=Path)
    parser.add_argument('--build-dir',type=Path)
    args=parser.parse_args()
    while not (OUT/'completed.json').exists():
        if (OUT/'failure.json').exists(): raise RuntimeError((OUT/'failure.json').read_text())
        time.sleep(5)
    for path in ['hypocotyl/code/summarize_observations.py','paper/write_hypocotyl_results.py',
                 'hypocotyl/code/package_release.py']:
        print(f'Running {path}',flush=True)
        subprocess.run([sys.executable,str(ROOT/path)],cwd=ROOT,check=True)
    if args.tectonic is None: return
    assert args.build_dir is not None
    compiler=args.tectonic.resolve(); build=args.build_dir.resolve(); build.mkdir(parents=True,exist_ok=True)
    paper=ROOT/'paper'
    result=subprocess.run([str(compiler),'--keep-logs','--keep-intermediates','--outdir',str(build),'main.tex'],
                          cwd=paper,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (build/'build-output.txt').write_text(result.stdout)
    print(result.stdout,flush=True)
    result.check_returncode()
    log=(build/'main.log').read_text(errors='replace')
    assert not re.search(r'(?:Citation|Reference) .*?undefined',log)
    assert 'There were undefined' not in log
    shutil.copyfile(build/'main.pdf',paper/'main.pdf')
    subprocess.run(['pdftotext','-layout',str(paper/'main.pdf'),str(build/'main.txt')],check=True)
    text=(build/'main.txt').read_text()
    source=(paper/'main_standalone.tex').read_text()
    assert 'effectively tied' not in text
    assert not any(key in source for key in ['you2026hypocotylpreprint','hypocotyl_light_interpolation','time_holdout'])
    assert all(key in source for key in ['alimchandani2026atlas','nozue2007rhythmic','seaton2015linked'])
    with zipfile.ZipFile(paper/'overleaf_20260909.zip') as z:
        assert z.read('main.tex')==(paper/'main_standalone.tex').read_bytes()
        assert z.read('references.bib')==(paper/'references.bib').read_bytes()
        for name in z.namelist():
            if name.startswith('figures/'): assert z.read(name)==(paper/name).read_bytes()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pages=text.split('\f')
    record=dict(status='compiled_and_text_checks_passed',target_type='individual_observed_cells',
        pdf_sha256=digest(paper/'main.pdf'),source_sha256=digest(paper/'main_standalone.tex'),
        figure_sha256=digest(paper/'figures/hypocotyl_light_replicates.pdf'),
        pages=len(pages)-(not pages[-1].strip()),
        relevant_pages=[i+1 for i,t in enumerate(pages) if any(s in t for s in
            ['Hypocotyl growth: individual','Hypocotyl replicate-group holdout:',
             'Hypocotyl predictions for Col-0'])],
        references_resolved=True,overleaf_bundle_verified=True,visual_review_completed=False,
        manuscript_uses_raw_observation_metrics=True,mean_target_comparison_paragraphs_removed=True,
        partition_provenance_retained_in_methods=True,
        latex_warnings=[s for s in result.stdout.splitlines() if 'warning:' in s.lower()])
    (paper/'compile_report_20260909.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__': main()
