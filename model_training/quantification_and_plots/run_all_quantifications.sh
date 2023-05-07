#!/bin/bash


# EfficientNetB4 experiments 
for e in 9 7 5 3 1
do
    for i in {1..10}
    do
        echo ">>>>>>>>> Running EfficientNetB4 ==> Experiment id $e, Iteration $i."
        python classification_quantification.py \
         -p /space/ariyanzarei/charcoal_dry_rot/datasets/patches/2022-04-18_512X512/test/images/ \
         -i /space/ariyanzarei/charcoal_dry_rot/raw_data/images/ \
         -l /space/ariyanzarei/charcoal_dry_rot/raw_data/masks/ \
         -o /space/ariyanzarei/charcoal_dry_rot/results/ \
         -e /space/ariyanzarei/charcoal_dry_rot/experiments/patch_size_classification_experiments_config_and_results_file.csv \
         -n $e
    done
done

# FCN (with Dice) experiments 
for e in 8 6 4 2 0
do
    for i in {1..10}
    do
        echo ">>>>>>>>> Running FCN ==> Experiment id $e, Iteration $i."
        python segmentation_quantification.py \
        -p /space/ariyanzarei/charcoal_dry_rot/datasets/patches/2022-04-18_512X512/test/images/ \
        -i /space/ariyanzarei/charcoal_dry_rot/raw_data/images/ \
        -l /space/ariyanzarei/charcoal_dry_rot/raw_data/masks/ \
        -o /space/ariyanzarei/charcoal_dry_rot/results/ \
        -e /space/ariyanzarei/charcoal_dry_rot/experiments/patch_size_segmentation_experiments_config_and_results_file.csv \
        -n $e
    done
done