#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv/cv.hpp"
#include "opencv2/highgui.hpp"
#include <QString>
#include <vector>
#include "highgui.hpp"
#include <QFileDialog>
#include <iostream>
#include <QFile>
#include <QTextStream>
#include <imagepreprocessing.h>
#include <aam.h>
#include "classifier.h"

using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
extern int mouse_x;
extern int mouse_y;
extern bool draw;
void my_mouse_callback( int event, int x, int y, int flags, void* param );
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_btnOpenImage_clicked();

    void on_btnSaveImage_clicked();

    void on_loadDataBtn_clicked();

    void on_btnGenerateModel_clicked();

    void on_btnRecognizeMood_clicked();

    void on_btnSaveMood_clicked();

    void on_btnSaveModel_clicked();

    void on_btnLoadModel_clicked();

private:
    Ui::MainWindow *ui;
    void showImage(Mat img);
    void showImageWindow(Mat img, QString name);
    void prepareWindowToDraw(string windowName, Mat image);
    void savePoints(int x, int y);
    Mat faceImage;
    Mat faceGray;
    vector< vector <int> > dataSet;
    vector<Mat> images;
    int imageNum;
    AAM aam;
    vector<string> moods;
    vector<int> faceMoods;
    Classifier classifier;
};

#endif // MAINWINDOW_H
