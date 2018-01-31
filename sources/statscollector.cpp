#include "headers/statscollector.hpp"
#include "headers/application.hpp"
#include <stdexcept>

Stats::StatsCollector::StatsCollector(const std::string& CSVFileNameRes, const std::string& CSVFileNameImg)
: mCSVRes(CSVFileNameRes + ".csv"), mCSVImg(CSVFileNameImg + ".csv")
{
    mCSVRes << "Teach index" << "Mean" << "MeanDis" << "Deviation" << "Confidence Range" << "" << "";
}

Stats::ErrorCollector& Stats::StatsCollector::operator[](unsigned int teachIndex)
{
    if(teachIndex > mErrorStats.size())
        throw std::logic_error("StatsCollector::operator[] - Erreur : Indice d'apprentissage trop grand");

    if(teachIndex == mErrorStats.size())
        mErrorStats.push_back(ErrorCollector());

    return  mErrorStats[teachIndex];
}

void Stats::StatsCollector::exportData(bool mustProcessData)
{
    if(!mustProcessData)
        throw std::logic_error("Not implemented yet");

    for (unsigned int index{0}; index < mErrorStats.size(); ++index)
    {
        ErrorCollector::StatisticData data{mErrorStats[index].processData()};

        mCSVRes << index << data.mean << data.meanDis << data.deviation << data.confidenceRange << endrow;
    }
}

void Stats::StatsCollector::exportImage(Eigen::MatrixXf image, unsigned int teachIndex)
{
    mCSVImg << teachIndex << endrow;
    for(int j(0); j<784; j++)
    {
        mCSVImg << image(j);
        if (j%28 == 27) mCSVImg << endrow;
    }
    mCSVImg << endrow;
}


csvfile* Stats::StatsCollector::getCSVFile()
{
    return &mCSVRes;
}

