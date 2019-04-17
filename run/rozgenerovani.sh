a=c100k_LLL_TTT_SSS.shc
for S in cos jw jwxcos; do
    sed -e "s/SSS/$S/" $a > ${a/SSS/$S};
done
