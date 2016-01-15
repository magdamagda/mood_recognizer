#include "classifier.h"

Classifier::Classifier()
{
}

Classifier::Classifier(Mat trainingSet, vector<int> trainingLabels)
{
    this->train(trainingSet, trainingLabels);
}

void Classifier::train(Mat trainingSet, vector<int> trainingLabels)
{
    Mat results=matFromVector(trainingLabels);
    //this->classifier.create();
    this->classifier=StatModel::train<NormalBayesClassifier>(trainingSet, ROW_SAMPLE, results, NormalBayesClassifier::Params());
}

float Classifier::predict(vector<int> data)
{
    Mat matData=matFromVector(data).t();
    matData.convertTo(matData, CV_32F);
    return this->classifier->predict(matData);
}

Mat Classifier::matFromVector(vector<int> v)
{
    Mat result(0, 0, CV_32S);
    for(int i=0; i<v.size(); i++)
    {
        result.push_back(v[i]);
    }
    return result;
}

void Classifier::save(QString file)
{
    classifier->save(file.toStdString());
}

void Classifier::load(QString file)
{
    classifier=StatModel::load<NormalBayesClassifier>(file.toStdString());
}
