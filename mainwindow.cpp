#include "mainwindow.h"
#include "ui_mainwindow.h"

namespace Ui
{
bool draw=false;
int mouse_x=0;
int mouse_y=0;

void my_mouse_callback( int event, int x, int y, int flags, void* param )
{
    //cout<<"Mouse event!"<<endl;
    mouse_x = x;
    mouse_y = y;
    if ( event == EVENT_LBUTTONDBLCLK )
    {
        draw = true;
        cout<<"Mouse clicked!"<<endl;
    }
    else
    {
        draw = false;
    }
}
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->imageNum=-1;
    this->moods.push_back("neutral");
    this->moods.push_back("happy");
    this->moods.push_back("sad");
    this->moods.push_back("suprised");
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btnOpenImage_clicked()
{
    QString fileName=QFileDialog::getOpenFileName(this, tr("Load image"), "/home/magda/training", tr("jpg (*.jpg)"));
    if(fileName!=NULL)
    {
        this->imageNum++;
        this->dataSet.push_back(vector<int>());
        Mat image = imread(fileName.toStdString(), 1);
        ImagePreProcessing processing(200, 200);
        Mat face=processing.prepareImageToAAM(image);
        this->images.push_back(face);
        QString windowname="image";//+QString::number(this->imageNum);
        showImageWindow(face, windowname);
        prepareWindowToDraw(windowname.toStdString(), face);
    }
}

void MainWindow::showImage(Mat img)
{
    QImage qimage((uchar*)img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
    //ui->image->setPixmap(QPixmap::fromImage(qimage));
}

void MainWindow::showImageWindow(Mat img, QString name)
{
    namedWindow(name.toStdString());
    imshow(name.toStdString(), img);
}

void MainWindow::prepareWindowToDraw(string windowName, Mat image)
{
    cout<<"prepare window to draw"<<endl;
    setMouseCallback(windowName, Ui::my_mouse_callback);
    Mat imageToDraw;
    image.copyTo(imageToDraw);
    while(true)
    {
        while ( waitKey(1) )
        {
            if ( Ui::draw )
            {
                circle(imageToDraw, Point(Ui::mouse_x, Ui::mouse_y), 1, Scalar(255, 0, 0), 1, 8);
                //int greyness=image.at<uchar>(Ui::mouse_y, Ui::mouse_x);
                cout<<Ui::mouse_x<<"   "<<Ui::mouse_y<<"  "<<endl;
                Ui::draw = false;
                savePoints(Ui::mouse_x, Ui::mouse_y);
                imshow(windowName, imageToDraw);
            }
        }
    }
}

void MainWindow::savePoints(int x, int y)
{
    this->dataSet[this->imageNum].push_back(x);
    this->dataSet[this->imageNum].push_back(y);
}

void MainWindow::on_btnSaveImage_clicked()
{
    QString fileName=QFileDialog::getExistingDirectory(this, tr("Save data"));
    QFile file(fileName+"/data");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
    QTextStream out(&file);
    out<<dataSet.size()<<endl;
    out<<dataSet[0].size()<<endl;
    for(int i=0; i<dataSet.size(); i++)
    {
        for(int j=0; j<dataSet[i].size(); j++)
        {
            out << dataSet[i][j]<<"\t";
        }
        out<<endl;
    }
    if(!QDir(fileName+"/images").exists())
    {
        QDir().mkdir(fileName+"/images");
    }
    for(int i=0; i<images.size(); i++)
    {
        QString imageFileName=fileName+"/images/aam" + QString::number(i) + ".jpg";
        imwrite(imageFileName.toStdString(), images[i]);
    }
    QFile mfile(fileName+"/moods");
    if (!mfile.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
    QTextStream mout(&mfile);
    mout<<dataSet.size()<<endl;
    mout<<dataSet[0].size()<<endl;
    for(int i=0; i<dataSet.size(); i++)
    {
        for(int j=0; j<dataSet[i].size(); j++)
        {
            mout << dataSet[i][j]<<"\t";
        }
        if(i<this->faceMoods.size())
            mout<<this->faceMoods[i];
        mout<<endl;
    }
}

void MainWindow::on_loadDataBtn_clicked()
{
    QString fileName=QFileDialog::getExistingDirectory(this, tr("Load data"));
    cout<<fileName.toStdString()<<endl;
    this->dataSet.clear();
    this->faceMoods.clear();
    this->imageNum=-1;
    this->images.clear();
    QFile file(fileName + "/moods");
    cout<<"file open"<<endl;
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;
    QTextStream in(&file);
    int dataSize, pointsNum;
    in>>dataSize>>pointsNum;
    this->imageNum=dataSize-1;
    for(int i=0; i<dataSize; i++)
    {
        this->dataSet.push_back(vector<int>());
        for(int j=0; j<pointsNum; j++)
        {
            int coordinates;
            in>>coordinates;
            this->dataSet[i].push_back(coordinates);
        }
        int face_mood;
        in>>face_mood;
        this->faceMoods.push_back(face_mood);
        QString imageFileName=fileName+"/images/aam" + QString::number(i) + ".jpg";
        Mat image = imread(imageFileName.toStdString(), 1);
        this->images.push_back(image);
    }
    cout<<"data loaded success "<<dataSize<<endl;
    showImageWindow(images[0], "first");
    showImageWindow(images[1], "second");
    for(int i=0; i<this->dataSet.size(); i++)
    {
        for(int j=0; j<this->dataSet[i].size(); j++)
        {
            cout<<"data set: "<<this->dataSet[i][j]<<endl;
        }
    }


}

void MainWindow::on_btnGenerateModel_clicked()
{
    Mat shape=Mat::eye(this->dataSet.size(),this->dataSet[0].size(), CV_64F);;
    for(int i=0; i<this->dataSet.size(); i++)
    {
        for(int j=0; j<this->dataSet[i].size(); j++)
        {
            shape.at<double>(i, j)=this->dataSet[i][j];
            cout<<"data set: "<<this->dataSet[i][j]<<endl;
        }
    }
    cout<<"data set transformed:  "<<shape<<endl;
    //AAM test(200, 200);
    aam.train(this->images, shape);
    //test.train(this->images, shape);
    shape.convertTo(shape, CV_32S);
    this->classifier.train(shape, this->faceMoods);

}

void MainWindow::on_btnRecognizeMood_clicked()
{
    QString fileName=QFileDialog::getOpenFileName(this, tr("Load image"), "/home", tr("jpg (*.jpg)"));
    if(fileName!=NULL)
    {
        Mat image = imread(fileName.toStdString(), 1);
        ImagePreProcessing processing(200, 200);
        Mat face=processing.prepareImageToAAM(image);
        QString windowname="imageToRecognize";//+QString::number(this->imageNum);
        showImageWindow(face, windowname);
        vector<Point> result=aam.findPoints(face);
        //cout<<result<<endl;
        vector<int> data;
        for(int i=0; i<result.size(); i++)
        {
            data.push_back(result[i].x);
            data.push_back(result[i].y);
        }
        float mood=this->classifier.predict(data);
        cout<<mood<<endl;
        ui->labelMood->setText(ui->moods->itemText(mood));
    }
}

void MainWindow::on_btnSaveMood_clicked()
{
    this->faceMoods.push_back(this->ui->moods->currentIndex());
}

void MainWindow::on_btnSaveModel_clicked()
{
    QString fileName=QFileDialog::getExistingDirectory(this, tr("Save model"));
    this->aam.saveModel(fileName);
    this->classifier.save(fileName + "/classifier");
}

void MainWindow::on_btnLoadModel_clicked()
{
    QString fileName=QFileDialog::getExistingDirectory(this, tr("Save model"));
    this->aam.loadModel(fileName);
    this->classifier.load(fileName + "/classifier");
}
