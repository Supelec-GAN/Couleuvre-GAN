#include "headers/errorcollector.hpp"

#include <numeric>
#include <algorithm>
#include <math.h>

Stats::ErrorCollector::ErrorCollector()
: mErrorsGen(),
mErrorsDis()
{

}

Stats::ErrorCollector::StatisticData Stats::ErrorCollector::processData() const
{
    StatisticData data;

    // Calcul de la moyenne
    data.meanGen = std::accumulate(mErrorsGen.begin(), mErrorsGen.end(), 0.f)/(static_cast<float>(mErrorsGen.size()));
    data.meanDis = std::accumulate(mErrorsDis.begin(), mErrorsDis.end(), 0.f)/(static_cast<float>(mErrorsGen.size()));
	
    // Calcul d'écart type
    float deviationGen{0};
	float deviationDis{0};

    if(mErrorsGen.size() != 1)
    {
        std::for_each(mErrorsGen.begin(), mErrorsGen.end(), [&] (float x) {deviationGen += pow(x-data.meanGen, 2);});
        data.deviationGen = sqrt(deviationGen/static_cast<float>(mErrorsGen.size()-1));
    }
	if(mErrorsDis.size() != 1)
	{
		std::for_each(mErrorsDis.begin(), mErrorsDis.end(), [&] (float x) {deviationDis += pow(x-data.meanDis, 2);});
		data.deviationDis = sqrt(deviationDis/static_cast<float>(mErrorsDis.size()-1));
	}
    // Calcul d'interval de confiance
    data.confidenceRangeGen = 2*data.deviationGen/(sqrt(static_cast<float>(mErrorsGen.size())));
	data.confidenceRangeDis = 2*data.deviationDis/(sqrt(static_cast<float>(mErrorsDis.size())));

    return data;
}

void Stats::ErrorCollector::addResultGen(float result)
{
    mErrorsGen.push_back(result);
}

void Stats::ErrorCollector::addResultDis(float result)
{
    mErrorsDis.push_back(result);
}
