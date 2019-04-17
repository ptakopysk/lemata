for p in $(cat lang_name)
do
    l=${p%_*}; t=${p#*_}
    for f in $(ls *LLL*|grep -v SSS)
    do
        g=${f/LLL/$l}
        h=${g/TTT/$t}
        sed -e "s/LLL/$l/g" -e "s/TTT/$t/g" $f > $h
    done
done
