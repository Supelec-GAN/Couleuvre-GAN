# Script gnuplot d'affichage des courbes

set datafile separator ";"
set term png
system 'mkdir -p graphs'
set output "graphs/graph.png"

set xlabel 'Nb Apprentissages' # A multiplier par le nombre d'apprentissage par index
set ylabel 'Score'

set key out vert
set key right top
set key autotitle columnheader
set key box 3 
set key width 1

set yrange[0:1]

plot for [i=2:3] 'resultat.csv' using 1:i with linespoints
