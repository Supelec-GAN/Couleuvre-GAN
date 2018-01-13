#ifndef STATSCOLLECTOR_HPP
#define STATSCOLLECTOR_HPP

#include "headers/errorcollector.hpp"
#include "headers/CSVFile.h"
#include <eigen3/Eigen/Dense>

#include <vector>
#include <string>


namespace Stats
{

class StatsCollector
{
    public:
        StatsCollector(const std::string& CSVFileNameRes = "resultat", const std::string& CSVFileNameImg = "image");

        ErrorCollector& operator[](unsigned int teachIndex);

        void exportData(bool mustProcessData = true);

        void exportImage(Eigen::MatrixXf image, unsigned int teachIndex);

        csvfile* getCSVFile();

    private:
        std::vector<ErrorCollector> mErrorStats;
        csvfile                     mCSVRes;
        csvfile                     mCSVImg;
};

}

#endif // STATSCOLLECTOR_HPP
