#!/bin/bash

# Launch graspnet_server in a new terminal
gnome-terminal -- bash -c " \
echo 'Activating graspnet env'; \
source /home/edward/miniconda3/etc/profile.d/conda.sh; \
conda activate contact_graspnet_env; \
cd ~/Imperial/fyp-robotGPT/; \
echo 'Launching graspnet_server'; \
python src/fyp_package/graspnet_server.py; \
exec bash"

# Launch langsam_server in a new terminal
gnome-terminal -- bash -c " \
echo 'Activating langsam env'; \
source /home/edward/miniconda3/etc/profile.d/conda.sh; \
conda activate langsam_env; \
cd ~/Imperial/fyp-robotGPT/; \
echo 'Launching langsam_server'; \
python src/fyp_package/langsam_server.py; \
exec bash"
