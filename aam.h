#ifndef AAM_H
#define AAM_H
#include "opencv/cv.hpp"
#include "opencv2/highgui.hpp"
#include <QString>
#include <vector>
#include "highgui.hpp"
#include <QFile>
#include <QTextStream>
#include <imgproc.hpp>
#include <map>
#include <comparepoint.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace cv;

class AAM
{
public:
    AAM();
    AAM(int w, int h, int sampling, int iterations, float maxError);
    int pointsNumber;
    void addExample(Mat shape, Mat image);
    void train(vector<Mat> images, Mat shapes);
    vector<Point> findPoints(Mat image);
    void saveModel(QString dir);
    void loadModel(QString dir);

private:
    vector<Mat> images;
    Mat A;
    int ImageHeight;
    int ImageWidth;
    Mat createConvexHull(vector<Point> points);
    Mat warpImage(Mat image, vector<int> points);
    void addShape(Mat shape);
    void addTexture(Mat texture);
    void addImage(Mat image);
    Mat getTextureInsideHull(Mat image, Mat convexHull);
    void saveMat(QString filename, Mat mat);
    Mat loadMat(QString filename);
    Subdiv2D triangulatePoints(vector<Point> points);
    Mat findTransformationMatrix(Point2f from[], Point2f to[]);
    void findShapeTextureDependency();
    Mat findLinearAproximation(Mat X, Mat Y);
    Mat countVariationMatrix(Mat matrix, Mat mean);
    PCA shapePCA;
    PCA texturePCA;
    Mat shapeSet;
    Mat textureSet;
    Mat meanShape;
    vector<Point> meanPoints;
    Mat meanTexture;
    Subdiv2D meanShapeTriangulation;
    map<Point, unsigned int, comparePoint> meanShapePointsOrder;
    double euqlidesNorm(vector<float> w);
    int E;
    int iterations;
    void displayTriangulation(Subdiv2D subdiv);
    bool isPointOnImage(Point point, int w, int h);
    void displayModel();
    vector<Point> convertMatToPoints(Mat points);
    void drawPoints(Mat image, Mat points, string name);
    void countA();
    Mat randomPointsMat(int max, int size);
    void makeShapeModel();
    Mat deltaS;
    void triangulateMeanPoints();
    void makeTextureModel();
    Mat b_g;
    void getW();
    Mat W;
    double w;
    void makeApearanceModel();
    Mat appearance;
    PCA appearancePCA;
    Mat cSet;
    Mat sampleMat(Mat mat, int x);
    Mat getShapeParamsFromC(Mat c);
    Mat getTextureParamsFromC(Mat c);
    Mat getTetureInShape(Mat image, Mat shape);
    Mat meanConvexHull;
    PCA textureVariationPCA;
    Mat cutToRange(Mat mat, int min, int max);
    int sampling;
};

#endif // AAM_H
