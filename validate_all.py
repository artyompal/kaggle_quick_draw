#!/usr/bin/python3.6

import os, sys, subprocess
from glob import glob
from shutil import copyfile

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <directory>')
        sys.exit()

    for orig_name in sorted(glob(os.path.join(sys.argv[1], '*.pk'))):
        print(f'processing {orig_name}')

        if orig_name.find('_best_[') != -1:
            print('not original, skipping')
            continue

        name = orig_name[:orig_name.rfind('_')]
        print('name', name)

        subprocess.run(["./validate.py", orig_name])

        with open('../models/validation/log_validation.txt') as f:
            s = [s for s in f][-1]
            map3 = s.split(' ')[-1].rstrip()
            test_float = float(map3)

        new_name = f'{name}_{map3}.pk'
        backup_name = f'{orig_name[:-3]}.bak'

        print('new_name', new_name, 'backup_name', backup_name)
        copyfile(orig_name, new_name)
        os.rename(orig_name, backup_name)

