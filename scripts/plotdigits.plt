set datafile separator ";"
set term png
set palette defined (0 "black", 1 "white")

do for [n=0:1] {
    set output sprintf("digit_%d.png", n)
    plot "exemple.dat" index n matrix with image
}
