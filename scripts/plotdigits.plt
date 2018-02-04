# Lecture des arguments du script
if(!exists("datafile")) datafile='exemple_digits.dat'

# Paramétrage du script
set datafile separator ";"
set term png
set palette defined (0 "black", 1 "white")
system 'mkdir -p digits'

#Affichage
do for [n=0:1] {
    set output sprintf("digits/digit_%d.png", n)
    plot datafile index n matrix with image
}
