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
            float mean;
            float deviation;
            float confidenceRange;
        };

    public:
                        ErrorCollector();

        StatisticData   processData() const;
        void            addResult(float result);

    private:


    private:
        std::vector<float> mErrors;

};

}

#endif // ERRORCOLLECTOR_HPP
