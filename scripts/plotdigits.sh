#!/usr/bin/bash

imageFile="images.csv"

# On récupère le nombre d'images qu'on va vouloir plot
nbImages=$(((`grep -c '#' $imageFile`)-1))

# On supprime les points virgules finaux de chaque ligne (sinon ca déconne)
sed -i "s/;$//" $imageFile

# On supprime tout élément précédant un dièse (les dièses indiquent les commentaires mais parfois le csv est mal écrit)
sed -i "s/.*#/#/" $imageFile

# On créé le répertoire de sauvegarde des images s'il n'existe pas déjà
mkdir -p digits

# On appelle le script gnuplot avec les paramètres
gnuplot -e "nbImages=$nbImages" plotdigits.plt
