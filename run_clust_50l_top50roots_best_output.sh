for r in $(cat derroot_lemma_forms_50l.roots.top50)
do
    # simpl
    for s in jwxcos
    do
        qsub -N clust-pairs-simpl-$s-$(echo $r|unidecode) analyze_lc_50l_ROOT_SIM_PARAMS.shc $r $s -S -O -E pairs
        qsub -N anal-thresh-simpl-$s-$(echo $r|unidecode) analyze_lc_50l_ROOT_SIM_PARAMS.shc $r $s -S -O -T
        sleep 0.1s
    done
done
