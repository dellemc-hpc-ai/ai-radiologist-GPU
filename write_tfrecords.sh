# This script writes tf records into a folder called tf_records.
# Before you run this script please ensure you have the right conda env activated

#!/bin/bash 
mkdir -p tf_records

# write training tf records
python write_totfrec.py

# write validation tf records
python write_totfrec_val.py 


