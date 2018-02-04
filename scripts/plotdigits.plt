if(!exists("datafile")) datafile='exemple.dat'

set datafile separator ";"
set term png
set palette defined (0 "black", 1 "white")

do for [n=0:1] {
    set output sprintf("digits/digit_%d.png", n)
    plot datafile index n matrix with image
}
