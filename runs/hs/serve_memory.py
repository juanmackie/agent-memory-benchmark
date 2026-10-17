# Launcher with a neutral name: other sessions on this machine stop their servers with broad
# `pkill -f hindsight-api`, which also took this one down mid-experiment.
from hindsight_api.main import main

main()
