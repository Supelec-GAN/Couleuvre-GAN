#ifndef DATACOLLECTOR_H
#define DATACOLLECTOR_H

#include <vector>
#include <algorithm>

#include "headers/utility.hpp"
#include "headers/dataset.hpp"
#include "headers/CSVFile.h"

/// Classe servant à collecter et exporter les données dans un fichier .csv
class DataCollector
{
public:

    /// Constructeur permettant d'initialiser le DataCollector avec le nom du csv file
    /**
     * Initialisation du DataCollector contenant un csv file et les données voulues
     * @param csvName : nom du csv file associé à ce relevé de données
     * @param isErrorSet : génère les paires <clé, vecteur<float> vide> pour les donnees d'un DataSet : numéro de batch, erreur moyenne, écart-type et intervalle de confiance à 95%
     */
    DataCollector(std::string csvName = "resultat", bool isErrorSet = 0);

    /// Ajout d'un nouveau type de données sous la clé type
    void addDataType(std::string type);

    [[deprecated]]
    void addData(DataSet dataSet);
    /// Ajout d'un set de données traitées (abscisse, moyenne, écart-type, intervalle de confiance)
    void addProcessedData(DataSet dataSet);

    /// Ajout d'un set de données selon la clé "key" et les données brutes du DataSet
    void addRawData(std::string key, DataSet dataSet);

    /// Inscrit le vecteur de données dans le csv file
    void exportData();

    /// Surcouche de l'opérateur [] : fonctionne comme pour une map
    /**
    * dataCollector[key] renvoie le vecteur de données associé à la key
    * Si cette paire n'existe pas encore, elle est créée. Cela crée une nouvelle colonne de données
    */
    std::vector<float>& operator[] (std::string key);


private:
    /// Vecteur de données : vecteur de paires : nom de la donnée (clé) et ensemble des données associées
    /**
    * @param std::string : Type de donnée
    * @param std::vector<float> : vecteur contenant les données
    */
    std::vector<std::pair<std::string, std::vector<float>>> mDataVector;
    csvfile                                                 csv;
};

#endif // DATACOLLECTOR_H
