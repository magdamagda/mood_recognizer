#include "imagepreprocessing.h"

ImagePreProcessing::ImagePreProcessing():W(200), H(200)
{
}

ImagePreProcessing::ImagePreProcessing(int w, int h):W(w), H(h)
{
}

const string ImagePreProcessing::face_cascade_name = "haarcascade_frontalface_alt.xml";

Mat ImagePreProcessing::prepareImageToAAM(Mat image)
{
    Mat image_grey;
    cvtColor(image, image_grey, COLOR_BGR2GRAY);
    vector< Rect_<int> > faces = findFace(image_grey);
    cout<<"faces found"<<endl;
    if (faces.size()<=0)
    {
        return Mat::zeros(1,1, CV_8UC1);
    }
    Rect rec=faces[0];
//    rec.y=rec.y+rec.height/2;
//    rec.height=rec.height/2;
//    Mat face=image_grey(rec);
//    //imshow("face", face);
//    vector< Rect_<int> > mouths = findMouth(face);
//    Mat mouth=face(mouths[0]);
//    normalizeImage(mouth);
//    mouth=this->resize(mouth, H, W);
//    return mouth;
    Mat face=image_grey(rec);
    normalizeImage(face);
    face=this->resize(face, H, W);
    return face;
}

Mat ImagePreProcessing::prepareFaceToAAM(Mat face)
{
    Mat face_grey;
    cvtColor(face, face_grey, COLOR_BGR2GRAY);
    normalizeImage(face_grey);
    face_grey=this->resize(face_grey, H, W);
    return face_grey;
}

vector< Rect_<int> > ImagePreProcessing::findFace(Mat image)
{
    CascadeClassifier face_cascade;
    vector< Rect_<int> > faces;
    if(!face_cascade.load(face_cascade_name))
    {
        cout<<"error: Brak klasyfikatora twarzy!"<<endl;
    }
    else
    {
        face_cascade.detectMultiScale(image, faces);
    }
    return faces;
}

void ImagePreProcessing::normalizeImage(Mat image)
{
    normalize(image, image, 0, 255, NORM_MINMAX);
}

Mat ImagePreProcessing::resize(Mat image, int height, int width)
{
    Mat scaled_image=Mat::zeros(height, width, image.type());
    cv::resize(image, scaled_image, Size(width, height));
    return scaled_image;
}

vector<Rect_<int> > ImagePreProcessing::findMouth(Mat image)
{
    CascadeClassifier face_cascade;
    vector< Rect_<int> > faces;
    string mouth_cascade_name="Mouth.xml";
    if(!face_cascade.load(mouth_cascade_name))
    {
        cout<<"error: Brak klasyfikatora ust!"<<endl;
    }
    else
    {
        face_cascade.detectMultiScale(image, faces);
    }
    return faces;
}
