for r in kód
do
    # simpl
    for s in jwxcos
    do
        qsub -N clust-pairs-simpl-2-$s-$(echo $r|unidecode) analyze_lc_50l_ROOT_SIM_PARAMS.shc $r $s -S -O -E pairs
        #qsub -N anal-thresh-simpl-$s-$(echo $r|unidecode) analyze_lc_50l_ROOT_SIM_PARAMS.shc $r $s -S -O -T
        sleep 0.1s
    done
done
