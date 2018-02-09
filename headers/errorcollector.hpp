#ifndef ERRORCOLLECTOR_HPP
#define ERRORCOLLECTOR_HPP

#include <vector>

namespace Stats
{

class ErrorCollector
{
    public:
        struct StatisticData
        {
            float meanGen;
            float meanDis;
            float deviationGen;
            float confidenceRangeGen;
			float deviationDis;
			float confidenceRangeDis;

		};

    public:
                        ErrorCollector();

        StatisticData   processData() const;
        void            addResultGen(float result);
        void            addResultDis(float result);

    private:


    private:
        std::vector<float> mErrorsGen;
        std::vector<float> mErrorsDis;

};

}

#endif // ERRORCOLLECTOR_HPP
