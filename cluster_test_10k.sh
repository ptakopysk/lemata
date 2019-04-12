set -o xtrace
./cluster.py -n 10000  -s jwxcos fasttext.cs cs_pdt-ud-all.conllu cs_pdt-ud-dev.conllu $@
