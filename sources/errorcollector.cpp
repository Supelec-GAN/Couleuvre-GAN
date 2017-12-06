#include "headers/errorcollector.hpp"

#include <numeric>
#include <algorithm>
#include <math.h>

Stats::ErrorCollector::ErrorCollector()
: mErrors()
{

}

Stats::ErrorCollector::StatisticData Stats::ErrorCollector::processData() const
{
    StatisticData data;

    // Calcul de la moyenne
    data.mean = std::accumulate(mErrors.begin(), mErrors.end(), 0.f)/(static_cast<float>(mErrors.size()));

    // Calcul d'écart type
    float deviation{0};
    if(mErrors.size() != 1)
    {
        std::for_each(mErrors.begin(), mErrors.end(), [&] (float x) {deviation += pow(x-data.mean, 2);});
        data.deviation = sqrt(deviation/static_cast<float>(mErrors.size()-1));
    }

    // Calcul d'interval de confiance
    data.confidenceRange = 2*data.deviation/(sqrt(static_cast<float>(mErrors.size())));

    return data;
}

void Stats::ErrorCollector::addResult(float result)
{
    mErrors.push_back(result);
}
