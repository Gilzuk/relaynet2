import glob

with open('ch00_status.txt', 'w') as f:
    f.write('Local ch00 files:\n')
    for file in glob.glob('chapters/ch00*.tex'):
        f.write(f'  {file}\n')