#ifndef COMPAREPOINT_H
#define COMPAREPOINT_H
#include <opencv.hpp>

using namespace cv;

class comparePoint
{
public:
    comparePoint();
    bool operator()(Point x, Point y);
};

#endif // COMPAREPOINT_H
