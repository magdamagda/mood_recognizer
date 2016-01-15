#include "comparepoint.h"

comparePoint::comparePoint()
{
}

bool comparePoint::operator()(Point x, Point y)
{
    if (x.x<y.x)
    {
        return true;
    }
    else
    {
        if(x.x==y.x)
        {
            return x.y<y.y;
        }
        return false;
    }
}
