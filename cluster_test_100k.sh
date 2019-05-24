set -o xtrace
./cluster.py -n 100000  -s jwxcos fasttext.cs cs_pdt-ud-dev.conllu $@ -b -C
./cluster.py -n 100000  -s jwxcos fasttext.cs cs_pdt-ud-dev.conllu $@ -C
