#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include<opencv/ml.h>
#include "opencv/cv.hpp"
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include <stdfix.h>
#include <QString>

using namespace std;
using namespace cv;
using namespace cv::ml;

class Classifier
{
public:
    Classifier();
    Classifier(Mat trainingSet, vector<int> trainingLabels);
    void train(Mat trainingSet, vector<int> trainingLabels);
    float predict(vector<int> data);
    void save(QString file);
    void load(QString file);


private:
    Ptr<StatModel> classifier;
    Mat matFromVector(vector<int> v);
};

#endif // CLASSIFIER_H
