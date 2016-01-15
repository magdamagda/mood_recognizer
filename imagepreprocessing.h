#ifndef IMAGEPREPROCESSING_H
#define IMAGEPREPROCESSING_H
#include "opencv/cv.hpp"
#include "opencv2/highgui.hpp"
#include <QString>
#include <vector>
#include "highgui.hpp"
#include <QFile>
#include <QTextStream>
#include <imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class ImagePreProcessing
{
public:
    ImagePreProcessing();
    ImagePreProcessing(int w, int h);
    Mat prepareImageToAAM(Mat image);
    Mat prepareFaceToAAM(Mat faceImage);
    vector<Rect_<int> > findFace(Mat image);
    vector<Rect_<int> > findMouth(Mat image);

    const int W;
    const int H;
    static const string face_cascade_name;
    void normalizeImage(Mat image);
    Mat resize(Mat image, int height, int width);

};

#endif // IMAGEPREPROCESSING_H
