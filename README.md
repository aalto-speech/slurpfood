# SLURPFOOD

This repository contains the SLURPFOOD splits proposed in the paper "Out-of-distribution generalisation in spoken language understanding".
Additionally, it contains scripts for generating the splits and creating the baselines.

To start, you need to first clone the original [SLURP repository](https://github.com/pswietojanski/slurp) in the parent directory.

Then you need to download the SLURP audio files by navigating to `cd slurp/scripts` and running `./download_audio.sh`.


The `slurpfood` directory contains two subdirectories: `scripts` and `splits`. The `scripts` contains the scripts used for reproducing the splits and the `splits` contain the actual data splits in JSON and CSV formats.

In the `experiments` directory are the SpeechBrain recipes for training the models discussed in the paper.
