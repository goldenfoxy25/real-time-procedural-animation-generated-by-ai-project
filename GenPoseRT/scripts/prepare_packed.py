"""
Helper to prepare conversion commands for the 'Packed' dataset.

- Scans an input folder for .fbx files and .npy files.
- For each .fbx prints a PowerShell command line to run Blender with the conversion script
  `scripts/blender/convert_fbx_to_npy.py` and produce a corresponding .npy under output_dir.
- Optionally prints a Windows PowerShell snippet that runs conversions in sequence.

This script does not call Blender itself (Blender must be installed separately). It simply
creates reproducible commands the user can run.

Usage:
    # Example (repository-root relative):
    python scripts/prepare_packed.py --input "../Packed" --output "../data/converted_packed"
"""
import os
import argparse

DEFAULT_BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender\blender.exe"


def find_files(input_dir):
    fbx_files = []
    npy_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.fbx'):
                fbx_files.append(os.path.join(root, f))
            elif f.lower().endswith('.npy'):
                npy_files.append(os.path.join(root, f))
    return sorted(fbx_files), sorted(npy_files)


def make_command(blender_exe, script_path, in_path, out_path):
    be = f'"{blender_exe}"'
    sp = f'"{script_path}"'
    inp = f'"{in_path}"'
    outp = f'"{out_path}"'
    # pass folder (not single file) so the converter will scan that folder for .fbx
    cmd = f'{be} --background --python {sp} -- --input_dir {inp} --output_dir {outp}'
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--blender', default=DEFAULT_BLENDER_EXE)
    p.add_argument('--script', default=os.path.join(os.path.dirname(__file__), 'blender', 'convert_fbx_to_npy.py'))
    p.add_argument('--target_model', default=None, help='Optional path to a target skeleton JSON to recommend fine-tuning')
    args = p.parse_args()

    fbx_files, npy_files = find_files(args.input)
    print(f'Found {len(fbx_files)} FBX files and {len(npy_files)} existing NPY files under', args.input)
    cmds = []
    for f in fbx_files:
        rel = os.path.relpath(f, args.input)
        out_npy = os.path.join(args.output, os.path.splitext(rel)[0] + '.npy')
        os.makedirs(os.path.dirname(out_npy), exist_ok=True)
        cmd = make_command(args.blender, args.script, os.path.dirname(f), args.output)
        cmds.append((f, out_npy, cmd))

    if len(cmds) == 0:
        print('No FBX files to convert. If you have .npy files already, you can copy them to the output folder:')
        for n in npy_files:
            print('Copy:', n)
        return

    print('\nPowerShell commands to convert FBX folders to NPY (run these in PowerShell):\n')
    seen_dirs = set()
    for f, out, cmd in cmds:
        d = os.path.dirname(f)
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        print(cmd)

    print('\nNotes:')
    print('- Ensure Blender path is correct and the conversion script exists at scripts/blender/convert_fbx_to_npy.py')
    print('- On Windows, run PowerShell as Administrator if you get permission errors')
    print('- After conversion, use src.data_loader.PackedDataset to load the .npy files')
    if args.target_model:
        print('\nTarget model provided:', args.target_model)
        print('The converter will also produce per-animation .json metadata files (bone order, frame rate) which can be used to remap/retarget during training or fine-tuning.')

if __name__ == '__main__':
    main()
