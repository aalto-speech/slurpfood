To create the compositional generalisation (CG) splits, first clone the repository at [aalto-speech/dbca](https://github.com/aalto-speech/dbca) to [this folder](https://github.com/aalto-speech/slurpfood/tree/main/slurpfood/scripts/CG)
```
git clone git@github.com:aalto-speech/dbca.git
```

After that, prepare the data for the splitting algorithm:
```
compound_type=scenario_action
prepped_data_dir=prepped_data/${compound_type}

python split_data.py \
    --prepare-data \
    --prepped-data-dir $prepped_data_dir \
    --compound-type $compound_type
```

After the data has been prepared, it can be splitted into train/test sets. The data splitting was done using the greedy algorithm from `aalto-speech/dbca` in two phases including manual selection of an approprate iteration of the algorithm after the first phase. This is not the most elegant way to split the data; the selection of the iteration could be automated too.

To create the CG split and the contrasting non-CG split, we first created a test set that we use for both splits, by splitting the data with a compound divergence (comdiv) of 0.5:

```
# this is much faster with a GPU, but a small corpus like this takes only a minute with a good CPU too 
comdiv=0.5
splits_dir=splits/${compound_type}_comdiv${comdiv}

python split_data.py \
    --prepped-data-dir $prepped_data_dir \
    --compound-type $compound_type \
    --split-data \
    --comdiv $comdiv \
    --splitted-data-dir $splits_dir
```

After that, the test set is fixed, and two new training sets are created for the the CG and non-CG splits (the training set from the previous phase is not used). The iteration 9000 from the previous step is selected to create a test set of an approprate size while having good divergence values.
```
# split data with a pre-defined test set:
iter=9000
mkdir $splits_dir/iter${iter}
cp  $splits_dir/test_set_iter${iter}_* $splits_dir/iter${iter}
mv $splits_dir/iter${iter}/test_set* $splits_dir/iter${iter}/test.txt
touch $splits_dir/iter${iter}/train.txt

for comdiv in 0 1
do
    new_splits_dir=$splits_dir/iter${iter}/comdiv$comdiv
    presplit=$splits_dir/iter${iter}

    python split_data.py \
        --prepped-data-dir $prepped_data_dir \
        --compound-type $compound_type \
        --split-data \
        --comdiv $comdiv \
        --splitted-data-dir $new_splits_dir \
        --from-presplit $presplit
done
```

Then, select a good iteration of the splitting algorithm and convert data to the SpeechBrain format:
```
new_iter=8000
for comdiv in 0 1
do
    new_splits_dir=$splits_dir/iter${iter}/comdiv$comdiv

    python split_data.py \
        --prepare-sb-data \
        --splitted-data-ids $new_splits_dir/{train,test}_set_iter${new_iter}_*.txt \
        --train-dev-split 0.9 \
        --prepped-data-dir $prepped_data_dir \
        --sb-data-dir $new_splits_dir/iter${new_iter}
done
```

Finally, calculate the stats and divergences of the splits:
```
for comdiv in 0 1
do
    new_splits_dir=$splits_dir/iter${iter}/comdiv$comdiv
    
    python split_data.py \
        --analyse-splits \
        --sb-data-dir $new_splits_dir/iter${new_iter} \
        > $new_splits_dir/iter${new_iter}/stats.txt
done
```
The splits used in the Interspeech paper are in `slurpfood/slurpfood/splits/CG`, where "non_cg" splits refer to the comdiv=0 setup, and "cg" splits comdiv=1.