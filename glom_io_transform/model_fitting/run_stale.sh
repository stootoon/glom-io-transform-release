#!/bin/bash
source ../add_paths.sh
INDIR="$1"
NPERJOB=${2:-5}
JOBNAME=$(basename "$INDIR")
python check_dir.py $INDIR | shuf | xargs -n $NPERJOB | xargs -I% python $CFDGITPY/cmd2job.py 'python -u driver.py --run %' --jobtime "2-0" --jobname $JOBNAME --jobmem 48G --submit 
