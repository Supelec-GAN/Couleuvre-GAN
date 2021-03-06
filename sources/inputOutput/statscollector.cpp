#include "headers/inputOutput/statscollector.hpp"
#include "headers/application.hpp"
#include <stdexcept>

Stats::StatsCollector::StatsCollector(const std::string& CSVFileNameRes, const std::string& CSVFileNameImg)
: mCSVRes(CSVFileNameRes + ".csv"), mCSVImg(CSVFileNameImg + ".csv")
{
    mCSVRes << "Teach index" << "MeanGen" << "MeanDis" << "DeviationGen" << "ConfidenceRangeGen" << "DeviationDis" << "ConfidenceRangeDis" << "" << "";
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

        mCSVRes << index << data.meanGen << data.meanDis << data.deviationGen << data.confidenceRangeGen << data.deviationDis << data.confidenceRangeDis << endrow;
    }
}

void Stats::StatsCollector::exportImage(Eigen::MatrixXf image, unsigned int teachIndex, unsigned int sizeSide)
{
    mCSVImg << "#" << teachIndex << endrow;
    for(int j(0); j<sizeSide*sizeSide; j++)
    {
        mCSVImg << image(j);
        if (j%sizeSide == sizeSide - 1) mCSVImg << endrow;
    }
    mCSVImg << endrow;
}


csvfile* Stats::StatsCollector::getCSVFile()
{
    return &mCSVRes;
}

