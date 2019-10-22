# This script expects to have dataset downloaded. Please run download_dataset.sh
# before running this script.

#!/bin/bash 

mkdir -p images_all

for filename in ./tars/*.tar.gz
do 
  tar -xzvf $filename -C images_all/
done
 
