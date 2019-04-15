set -o xtrace
./cluster.py -n 1000  -s jwxcos fasttext.cs cs_pdt-ud-dev.conllu $@
