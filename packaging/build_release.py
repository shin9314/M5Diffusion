"""Build the local, ad-hoc signed alpha DMG; never includes model weights."""
import argparse, hashlib, json, os, plistlib, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSION='0.1.0-alpha'

def build(output):
    stage=ROOT/'work/release/dmg-stage'
    if stage.exists(): shutil.rmtree(stage)
    app=stage/'M5Diffusion.app'; contents=app/'Contents'; resources=contents/'Resources'; macos=contents/'MacOS'
    resources.mkdir(parents=True); macos.mkdir()
    runtime=ROOT/'work/release/runtime/python'
    manifest=json.loads((ROOT/'packaging/python-runtime.json').read_text())
    archive=ROOT/'work/release/python.tar.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['sha256']: raise RuntimeError('Python archive digest mismatch')
    shutil.copytree(runtime,resources/'runtime',symlinks=True)
    shutil.copytree(ROOT/'.venv/lib/python3.10/site-packages',resources/'runtime/lib/python3.10/site-packages',dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Entry-point scripts in the build venv can contain private interpreter paths.
    # Runtime invokes modules directly, never copied venv/bin scripts.
    payload=resources/'payload'; payload.mkdir()
    for name in ('m5diffusion','vendor'):
        shutil.copytree(ROOT/name,payload/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'serve.py',payload/'serve.py')
    shutil.copytree(ROOT/'assets/sd15-config',payload/'models/sd15-config',ignore=shutil.ignore_patterns('*.safetensors','*.bin','*.pth','*.ckpt'))
    shutil.copy2(ROOT/'packaging/bootstrap.py',resources/'bootstrap.py')
    for name in ('README.md','INSTALL.md','LICENSE','CHANGELOG.md','requirements-lock.txt'):
        shutil.copy2(ROOT/name,resources/name)
    shutil.copytree(ROOT/'docs/third-party',resources/'third-party',dirs_exist_ok=True)
    shutil.copy2(ROOT/'docs/third-party-licenses.md',resources/'THIRD-PARTY-LICENSES.md')
    plist={'CFBundleName':'M5Diffusion','CFBundleDisplayName':'M5Diffusion','CFBundleIdentifier':'org.m5diffusion.alpha','CFBundleExecutable':'M5Diffusion','CFBundlePackageType':'APPL','CFBundleShortVersionString':VERSION,'CFBundleVersion':'1','LSMinimumSystemVersion':'26.0','NSHighResolutionCapable':True}
    (contents/'Info.plist').write_bytes(plistlib.dumps(plist))
    subprocess.run(['xcrun','swiftc',str(ROOT/'packaging/Launcher.swift'),'-o',str(macos/'M5Diffusion'),'-target','arm64-apple-macos15.0','-framework','Cocoa'],check=True)
    for path in resources.rglob('*'):
        if path.is_symlink() and path.resolve().is_relative_to(ROOT/'.venv'): raise RuntimeError('Nonportable venv symlink')
        if path.is_file() and path.is_relative_to(payload) and path.suffix in ('.safetensors','.ckpt','.pth'): raise RuntimeError('Model weight included')
    subprocess.run(['codesign','--force','--deep','--sign','-',str(app)],check=True)
    (stage/'Applications').symlink_to('/Applications')
    shutil.copy2(ROOT/'INSTALL.md',stage/'INSTALL.md')
    output.mkdir(parents=True,exist_ok=True)
    dmg=output/f'M5Diffusion-v{VERSION}.dmg'
    subprocess.run(['hdiutil','create','-volname','M5Diffusion Alpha','-srcfolder',str(stage),'-ov','-format','UDZO',str(dmg)],check=True)
    digest=hashlib.sha256(dmg.read_bytes()).hexdigest()
    dmg.with_suffix('.dmg.sha256').write_text(digest+'  '+dmg.name+'\n')
    print(json.dumps({'app':str(app),'dmg':str(dmg),'sha256':digest,'developer_id_signed':False,'notarized':False}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();build(args.output)
