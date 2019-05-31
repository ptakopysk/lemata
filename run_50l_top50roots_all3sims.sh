for r in $(head -n 50 derroot_lemma_forms_50l.roots); do for s in cos jw jwxcos; do qsub -N pl-$s-$(echo $r|unidecode) plot_lc_50l_ROOT_SIM.shc $r $s; sleep 0.2s;done;done
