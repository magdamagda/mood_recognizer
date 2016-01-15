#include "aam.h"

AAM::AAM() : iterations(10), E(20), ImageHeight(200), ImageWidth(200), sampling(30)
{
    srand(time(0));
}

AAM::AAM(int w, int h, int sampling, int iterations, float maxError) : iterations(iterations), E(maxError), ImageHeight(h), ImageWidth(w), sampling(sampling)
{
    srand(time(0));
}

void AAM::addExample(Mat shape, Mat image)
{
    addShape(shape);
    addImage(image);
}

void AAM::addShape(Mat shape)
{
    shapeSet.push_back(shape);
}

void AAM::addTexture(Mat texture)
{
    textureSet.push_back(texture);
}

void AAM::addImage(Mat image)
{
    images.push_back(image);
}

void AAM::train(vector<Mat> images, Mat shapes)
{
    this->images=images;
    this->shapeSet = shapes;
    //cout<<"training shape set:  "<<shapeSet<<endl;
    makeShapeModel();
    triangulateMeanPoints();
    //cout<<"shape pca counted"<<endl;
    this->meanConvexHull = this->createConvexHull(this->meanPoints);
    cout<<"convexHull"<<meanConvexHull<<endl;
    for(int i=0; i<this->images.size(); i++)
    {
        Mat image_grey;
        cvtColor(this->images[i], image_grey, COLOR_BGR2GRAY);
        Mat texture=this->getTetureInShape(image_grey, this->shapeSet.row(i));
        //cout<<"texture smapled and normalized: "<<texture<<endl;
        this->addTexture(texture);
    }
    makeTextureModel();
    this->displayModel();
    getW();
    makeApearanceModel();
    this->countA();
    cout<<"A: "<<this->A<<endl;
    cout<<"AAM training complete"<<endl;

}

vector<Point> AAM::findPoints(Mat image)
{
    float k=10;
    int i=0;
    Mat convHull = this->createConvexHull(this->meanPoints);
    Mat points;
    cout<<"mean shape"<<this->meanShape<<endl;
    this->meanShape.copyTo(points);
    cout<<"initial shape"<<points<<endl;
    Mat deltaG;
    double normE0;
    Mat c;
    do
    {
        cout<<"iteration "<<i<<endl;
        cout<<"start Points: "<<points<<endl;
        k=1;
        //this->drawPoints(image, points, QString::number(i).toStdString());
        Mat textureSampled=this->getTetureInShape(image, points);
        textureSampled.convertTo(textureSampled, CV_32S);
        this->meanTexture.convertTo(meanTexture, CV_32S);
        Mat deltaT = textureSampled - this->meanTexture;
        points.convertTo(points, meanShape.type());
        Mat deltaPoint = points-meanShape;
       // cout<<"deltaPoint: "<<deltaPoint<<endl;
        deltaPoint=deltaPoint*this->w;
       // cout<<"deltaPoint: "<<deltaPoint<<endl;
        Mat b=this->textureVariationPCA.project(deltaT);
       // cout<<"b for texture: "<<b<<endl;
        Mat ap;
        deltaPoint.convertTo(deltaPoint, b.type());
        hconcat(deltaPoint, b, ap);
       // cout<<"appearance: "<<ap<<endl;
        c= this->appearancePCA.project(ap);
       // cout<<"c: "<<c<<endl;
        Mat modelAp=this->appearancePCA.backProject(c);
       // cout<<"model appearance: "<<modelAp<<endl;
        CvMat bModelMat;
        CvMat modelApMat=modelAp;
        cvGetCols(&modelApMat, &bModelMat, this->shapeSet.cols, this->shapeSet.cols + this->b_g.cols);
        Mat bModel=cvarrToMat(&bModelMat);
        bModel.convertTo(bModel, CV_32F);
       // cout<<"b model"<<bModel<<endl;
        Mat deltaGModel=this->textureVariationPCA.backProject(bModel);
       // cout<<"delta g model: "<<deltaGModel<<endl;
        deltaGModel.convertTo(deltaGModel, this->meanTexture.type());
        Mat textureModel=this->meanTexture + deltaGModel;
        textureModel.convertTo(textureModel, CV_8U);
        textureModel.convertTo(textureModel, CV_32S);
       // cout<<"texture model: "<<textureModel<<endl;
        Mat modelShape=this->getShapeParamsFromC(c);
        modelShape.convertTo(modelShape, this->meanShape.type());
        modelShape=meanShape + modelShape;
        modelShape=this->cutToRange(modelShape, 0, this->ImageHeight);
       // cout<<"model shape: "<<modelShape<<endl;
        Mat realTexture=this->getTetureInShape(image, modelShape);
        realTexture.convertTo(realTexture, textureModel.type());
       // cout<<"real texture"<<realTexture<<endl;
        Mat deltaI=realTexture-textureModel;
       // cout<<"deltaI: "<<deltaI<<endl;
        deltaI.convertTo(deltaI, CV_32F);
        Mat deltaC=deltaI*this->A;
        cout<<"delta c"<<deltaC<<endl;
        c.convertTo(c, deltaC.type());
        normE0=this->euqlidesNorm(deltaI);
        double normE=normE0;
        Mat newC;
        while(normE0<=normE && k>0.0001) //sprawdzić, czy deltaG jest mniejsze niż deltaG0
        {
           // cout<<"k "<<k<<endl;
            newC=c+k*deltaC;
            cout<<"newc: "<<newC<<endl;
            Mat shapeParams=this->getShapeParamsFromC(newC);
            Mat textureParams=this->getTextureParamsFromC(newC);
            Mat textureDelta=this->textureVariationPCA.backProject(textureParams);
            textureDelta.convertTo(textureDelta, meanTexture.type());
            Mat texture=meanTexture + textureDelta;
            shapeParams.convertTo(shapeParams, meanShape.type());
            points=meanShape + shapeParams;
            points=this->cutToRange(points, 0, this->ImageHeight);
            Mat realTexture=this->getTetureInShape(image, points);
            texture.convertTo(texture, realTexture.type());
            deltaI=realTexture - texture;
            normE=this->euqlidesNorm(deltaI);
            k=k/2;
        }
        i++;
        newC.copyTo(c);
        normE0=normE;
        cout<<"points: "<<points<<endl;
        cout<<"error: "<<normE0<<endl;
    }
    while(i<30); // sprwdzić, czy błąd deltaG jest mniejszy od maxymalnego błędu
    this->drawPoints(image, points, "final points");
    cout<<"final points: "<<points<<endl;
    cout<<"mean:         "<<this->meanShape<<endl;
    return this->convertMatToPoints(points);
}

double AAM::euqlidesNorm(vector<float> w)
{
    int sum=0;
    for(int i=0; i<w.size(); i++)
    {
        sum+=w[i]*w[i];
    }
    return sqrt(sum);
}

void AAM::saveMat(QString filename, Mat mat)
{
    mat.convertTo(mat, CV_32F);
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
    QTextStream out(&file);
    out<<mat.rows<<endl<<mat.cols<<endl;
    for(int i=0; i<mat.rows; i++)
    {
        for(int j=0; j<mat.cols; j++)
        {
            out << mat.at<float>(i, j)<<" ";
        }
        out<<endl;
    }
    file.close();
}

Mat AAM::loadMat(QString filename)
{
    cout<<"load mat"<<endl;
    Mat mat;
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return mat;
    QTextStream in(&file);
    int rows;
    int cols;
    in>>rows;
    in>>cols;
    cout<<rows<<"  "<<cols<<endl;
    for(int i=0; i<rows; i++)
    {
        Mat row;
        for(int j=0; j<cols; j++)
        {
            float item;
            in>>item;
            cout<<"item  "<<item<<endl;
            row.push_back(item);
        }
        cout<<"row: "<<i<<endl;
        row=row.t();
        mat.push_back(row);
        cout<<"row pushed"<<endl;
    }
    return mat;
}

Mat AAM::createConvexHull(vector<Point> points)
{
    Mat hull;
    convexHull(points, hull);
    return hull;
}

Mat AAM::warpImage(Mat image, vector<int> points)
{
    imshow("before warp", image);
    //cout<<"imege to warp type: "<<image.type()<<endl;
    Mat warp_final = Mat::zeros( image.rows, image.cols, image.type() );
    vector<Vec6f> trianglesList;
    this->meanShapeTriangulation.getTriangleList(trianglesList);
    for(int i=0; i<trianglesList.size(); i++)
    {
        Point p1(cvRound(trianglesList[i][0]), cvRound(trianglesList[i][1]));
        Point p2(cvRound(trianglesList[i][2]), cvRound(trianglesList[i][3]));
        Point p3(cvRound(trianglesList[i][4]), cvRound(trianglesList[i][5]));
        //cout<<"trinagle: "<<i<<"  "<<p1<<" "<<p2<<" "<<p3<<endl;
        if(isPointOnImage(p1, image.cols, image.rows) && isPointOnImage(p2, image.cols, image.rows) && isPointOnImage(p3, image.cols, image.rows))
        {
            Mat warp_mask = Mat::zeros( image.rows, image.cols, image.type() );
            Mat warp_dst = Mat::zeros( image.rows, image.cols, image.type() );
            //cout<<"p1: "<<p1<<endl;
           // cout<<"p2: "<<p2<<endl;
           // cout<<"p3: "<<p3<<endl;
            Point2f verticesDst[3];
            verticesDst[0]=p1;
            verticesDst[1]=p2;
            verticesDst[2]=p3;
            Point2f verticesSrc[3];
            int p1num=this->meanShapePointsOrder[p1];
            int p2num=this->meanShapePointsOrder[p2];
            int p3num=this->meanShapePointsOrder[p3];
            //cout<<"p1num: "<<p1num<<endl;
            //cout<<"p2num: "<<p2num<<endl;
            //cout<<"p3num: "<<p3num<<endl;
            verticesSrc[0]=Point(points[2*p1num], points[2*p1num+1]);
            verticesSrc[1]=Point(points[2*p2num], points[2*p2num+1]);
            verticesSrc[2]=Point(points[2*p3num], points[2*p3num+1]);
            //cout<<"dst: "<<verticesDst[0]<<" "<<verticesDst[1]<<" "<<verticesDst[2]<<endl;
            //cout<<"src: "<<verticesSrc[0]<<" "<<verticesSrc[1]<<" "<<verticesSrc[2]<<endl;
            Mat transformationMatrix=findTransformationMatrix(verticesSrc, verticesDst);
            //cout<<"transformation matrix: "<<transformationMatrix<<endl;
            warpAffine(image, warp_dst, transformationMatrix, warp_dst.size());
            imshow("warped", warp_dst);
            Point trianglePoints[3];
            trianglePoints[0]=verticesDst[0];
            trianglePoints[1]=verticesDst[1];
            trianglePoints[2]=verticesDst[2];
            fillConvexPoly(warp_mask, trianglePoints, 3, CV_RGB(255,255,255), CV_AA, 0 );
            //imshow("warp mask", warp_mask);
            warp_dst.copyTo(warp_final,warp_mask);
        }
    }
    imshow("warp final", warp_final);
    return warp_final;
}

Mat AAM::getTextureInsideHull(Mat image, Mat convexHull)
{
    //cout<<"type: "<<image.channels()<<" "<<image.type()<<endl;
    image.convertTo(image, 0);
    imshow("image to get texture", image);
    Mat texture(0, 1,  CV_64F);
    for(int i=0;i<image.rows;i++)
    {
        for (int j=0;j<image.cols;j++)
        {
            double distance = pointPolygonTest(convexHull,cvPoint2D32f(i,j),1);
            if(distance >=0){
                texture.push_back(image.at<uchar>(j,i));
            }
        }
    }
    return texture.t();
}

Subdiv2D AAM::triangulatePoints(vector<Point> points)
{
    Rect rect(0,0,this->ImageWidth, this->ImageHeight);
    Subdiv2D sub(rect);
    for(int i=0; i<points.size(); i++)
    {
        sub.insert(points[i]);
    }
    return sub;
}

void AAM::displayTriangulation(Subdiv2D subdiv)
{
    Mat img=Mat::zeros(ImageWidth, ImageHeight, CV_8UC1);
    Scalar delaunay_color=Scalar(255, 0, 0);
    bool draw;
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);

    for(int i = 0; i < triangleList.size(); i++)
    {
      Vec6f t = triangleList[i];

      pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
      pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
      pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

      draw=true;

      for(int i=0;i<3;i++)
      {
         if(pt[i].x>img.rows||pt[i].y>img.cols||pt[i].x<0||pt[i].y<0)
            draw=false;
      }
      if (draw)
      {
         line(img, pt[0], pt[1], delaunay_color, 1);
         line(img, pt[1], pt[2], delaunay_color, 1);
         line(img, pt[2], pt[0], delaunay_color, 1);
      }
    }
    imshow("triangulacja", img);
}

Mat AAM::findTransformationMatrix(Point2f from[3], Point2f to[3])
{
    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( from, to );
    return warp_mat;
}

void AAM::findShapeTextureDependency()
{
    Mat textureVariation =countVariationMatrix(this->textureSet, this->meanTexture);
    Mat shapeVariation = countVariationMatrix(this->shapeSet, this->meanShape);
    this->A = findLinearAproximation(textureVariation, shapeVariation);
}

Mat AAM::findLinearAproximation(Mat X, Mat Y)
{
    X.convertTo(X, CV_32F);
    Y.convertTo(Y, CV_32F);
    Mat result(Y.cols, X.rows, CV_32F);
   // Mat a=X.t()*X;
   // Mat b=X.t()*Y;

    solve(X, Y, result, DECOMP_QR);
    return result;
}

Mat AAM::countVariationMatrix(Mat matrix, Mat mean)
{
    matrix.convertTo(matrix, mean.type());
    Mat variationMatrix;
    for (int i=0; i<matrix.rows; i++)
    {
        Mat row=matrix.row(i) - mean;
        //cout<<"mean: "<<mean<<endl;
        //cout<<"row:  "<<matrixCopy.row(i)<<endl;
        //cout<<"resul:"<<row<<endl;
        variationMatrix.push_back(row);
    }
    return variationMatrix;
}

bool AAM::isPointOnImage(Point point, int w, int h)
{
    if(point.x>=w || point.x<0 || point.y>=h || point.y<0)
    {
        return false;
    }
    return true;
}

void AAM::displayModel()
{
    Mat convexHull = this->createConvexHull(this->meanPoints);
    Mat image=Mat::zeros(ImageWidth, ImageHeight, CV_8UC1);
    int n=0;
    //cout<<"Mean texture type "<<this->meanTexture.type()<<endl;
    for(int i=0;i<image.rows;i++)
    {
        for (int j=0;j<image.cols;j++)
        {
            double distance = pointPolygonTest(convexHull,cvPoint2D32f(i,j),1);
            if(distance >=0){
                if(n%30==0) {
                    cout<<this->meanTexture.at<int>(0, n/sampling)<<endl;
                    image.at<uchar>(j,i)=cvRound(this->meanTexture.at<int>(0, n/30));
                    //image.at<uchar>(j,i)=128;
                }
                    n++;
            }
        }
    }
    cout<<this->meanTexture.type()<<endl;
    imshow("mean model", image);
}

vector<Point> AAM::convertMatToPoints(Mat points)
{
    points.convertTo(points, CV_32S);
    vector<Point> result;
    for(int i=0; i<points.cols/2; i++)
    {
        result.push_back(Point(points.at<int>(2*i), points.at<int>(2*i+1)));
    }
    return result;
}

void AAM::drawPoints(Mat image, Mat points, string name)
{
    //cout<<"draw "<<name<<"  "<<points<<endl;
    Mat img;
    image.copyTo(img);
    vector<Point> vec=convertMatToPoints(points);
    cout<<vec<<endl;
    for(int i=0; i<vec.size(); i++)
    {
        circle(img, vec[i], 2, 255);
    }
    imshow(name, img);
}

void AAM::countA()
{
    Mat deltaC;
    Mat deltaI;
    for(int i=0; i<this->shapeSet.rows; i++)
    {

        for(int j=0; j<100; j++)
        {
            Mat ds=randomPointsMat(40, this->shapeSet.cols);
            cout<<"random ds: "<<ds<<endl;
            ds.convertTo(ds, this->shapeSet.type());
            Mat newShape=ds+this->shapeSet.row(i);
            newShape.convertTo(newShape, meanShape.type());
            Mat newShapeParams=newShape-meanShape;
            newShapeParams=newShapeParams*this->w;
            Mat image_grey;
            cvtColor(this->images[i], image_grey, COLOR_BGR2GRAY);
            Mat newTexture=this->getTetureInShape(image_grey, newShape);
            newTexture.convertTo(newTexture, meanTexture.type());
            Mat newDeltaTexture=newTexture - meanTexture;
            Mat newTextureParams=this->textureVariationPCA.project(newDeltaTexture);
            Mat newAp;
            newShapeParams.convertTo(newShapeParams, newTextureParams.type());
            hconcat(newShapeParams, newTextureParams, newAp);
            cout<<"new appearance: " <<newAp<<endl;
            Mat c=this->appearancePCA.project(newAp);
            cout<<"new c: "<<c<<endl;
            c.convertTo(c, cSet.type());
            Mat dc=this->cSet.row(i)-c;
            cout<<"dc: "<<dc<<endl;

            Mat ap = this->appearancePCA.backProject(c);
            CvMat apMat=ap;
            cout<<"appearance: "<<ap<<endl;
            CvMat shapeParams;
            cvGetCols(&apMat, &shapeParams, 0, shapeSet.cols);
            cout<<"shape: "<<endl;
            Mat shape=cvarrToMat(&shapeParams);
            cout<<"shapeParams: "<<shape<<endl;
            shape=shape/this->w;
            cout<<"shapeParams scaled: "<<shape<<endl;
            CvMat textureParams;
            cvGetCols(&apMat, &textureParams, shapeSet.cols, shapeSet.cols+this->b_g.cols);
            Mat texture=cvarrToMat(&textureParams);
            cout<<"texture Params: "<<texture<<endl;
            shape.convertTo(shape, meanShape.type());
            Mat modelShape=this->meanShape + shape;
            modelShape=this->cutToRange(modelShape, 0, this->ImageHeight);
            cout<<"modelShape: "<<modelShape<<endl;
            Mat backProjectShape=this->textureVariationPCA.backProject(texture);
            backProjectShape.convertTo(backProjectShape, meanTexture.type());
            Mat modelTexture=this->meanTexture + backProjectShape;
            modelTexture.convertTo(modelTexture, CV_8U);
            modelTexture.convertTo(modelTexture, CV_32S);
            //cout<<"modelTexture: "<<modelTexture<<endl;
            Mat warped=this->warpImage(this->images[i], modelShape);
            cvtColor(warped, image_grey, COLOR_BGR2GRAY);
            Mat convHull = this->createConvexHull(this->meanPoints);
            Mat realTexture=this->getTextureInsideHull(image_grey, convHull);
            Mat realTextureSampled=this->sampleMat(realTexture, sampling);
            normalize(realTextureSampled, realTextureSampled, 0, 255, NORM_MINMAX);
            cout<<realTexture.cols<<" "<<realTextureSampled.cols<<"  "<<modelTexture.cols<<endl;
            realTextureSampled.convertTo(realTextureSampled, modelTexture.type());
            Mat dI=realTextureSampled-modelTexture;
            deltaC.push_back(dc);
            deltaI.push_back(dI);
        }
    }
    cout<<"deltaI : " <<deltaI<<endl;
    cout<<"deltaC: "<<deltaC<<endl;
    this->A=this->findLinearAproximation(deltaI, deltaC);
    Mat deltaItest;
    deltaI.row(0).convertTo(deltaItest, A.type());
    cout<<"delta I row: "<<deltaItest.row(0)<<endl;
    cout<<"delta c row: "<<deltaC.row(0)<<endl;
    //cout<<this->A.type()<<endl;
    //cout<<deltaI.type()<<endl;
    Mat result=deltaItest.row(0)*A;
    cout<<result<<endl;
}

Mat AAM::randomPointsMat(int max, int size)
{

    Mat result=Mat::zeros(1, size, CV_32S);
    for(int i=0; i<size; i++)
    {
        result.at<int>(0, i) = rand()%(2*max)-max;
    }
    return result;
}

void AAM::makeShapeModel()
{
    shapePCA=PCA(shapeSet, Mat(), CV_PCA_DATA_AS_ROW);
    meanShape=shapePCA.mean;
    meanShape.convertTo(meanShape, CV_32S);
    this->deltaS=this->countVariationMatrix(shapeSet, meanShape);
    cout<<"shape set"<<shapeSet.type()<<endl;
   cout<<"mean shape: "<<meanShape<<endl;
}

void AAM::triangulateMeanPoints()
{
    cout<<"in triangulation mean function"<<endl;
    if (!this->meanShapePointsOrder.empty())
        this->meanShapePointsOrder.clear();
    for(int i=0; i<meanShape.cols/2; i++)
    {
        //pair<int, int> point=make_pair(meanShape.at<int>(0, 2*i), meanShape.at<int>(0, 2*i+1));
        Point point(meanShape.at<int>(0, 2*i), meanShape.at<int>(0, 2*i+1));
        this->meanShapePointsOrder[point]=i;
        this->meanPoints.push_back(Point(meanShape.at<int>(0, 2*i), meanShape.at<int>(0, 2*i+1)));
    }
    cout<<"mean points: "<<this->meanPoints<<endl;
    this->meanShapeTriangulation=triangulatePoints(this->meanPoints);
    this->displayTriangulation(this->meanShapeTriangulation);
    for (std::map<Point,unsigned int, comparePoint>::iterator it=this->meanShapePointsOrder.begin(); it!=this->meanShapePointsOrder.end(); ++it)
        std::cout << it->first << " => " << it->second << '\n';
}

void AAM::makeTextureModel()
{
    texturePCA(textureSet, Mat(), CV_PCA_DATA_AS_ROW);
    meanTexture=texturePCA.mean;
    meanTexture.convertTo(meanTexture, CV_32S);
    //cout<<"texture: "<<textureSet.row(0)<<endl;
    //cout<<"meanTexture: "<<meanTexture<<endl;
    Mat deltaG=this->countVariationMatrix(textureSet, meanTexture);
    //cout<<"deltaG: "<<deltaG.row(0)<<endl;
    deltaG.convertTo(deltaG, CV_32F);
    textureVariationPCA(deltaG, Mat(), CV_PCA_DATA_AS_ROW);
    this->b_g=textureVariationPCA.project(deltaG);
    //cout<<"bg: "<<b_g<<endl;
    //cout<<"backProject: "<<textureVariationPCA.backProject(b_g.row(0))<<endl;
}

void AAM::getW()
{
    /*for(int i=0; i<shapeSet.cols; i++)
    {
        vector<float> deltaG;
        for(int j=0; j<shapeSet.rows; j++)
        {

        }
    }*/
    /*Scalar sum1=sum(textureVariationPCA.eigenvalues);
    Scalar sum2=sum(shapePCA.eigenvalues);
    cout<<"texture eigen values: "<<textureVariationPCA.eigenvalues<<endl;
    cout<<"shape eigen values: "<<shapePCA.eigenvalues<<endl;
    Mat deltaXMat=this->shapeSet.row(0)-this->shapeSet.row(1);
    double deltaX=this->euqlidesNorm(deltaXMat);
    Mat deltaGMat=this->textureSet.row(0)-this->textureSet.row(1);
    double deltaG=this->euqlidesNorm(deltaGMat);
    //this->w=deltaG/deltaX;
    this->w=sum1[0]/sum2[0];*/
    Mat ds=this->randomPointsMat(5, this->shapeSet.cols);
    ds.convertTo(ds, shapeSet.type());
    Mat newShape= this->shapeSet.row(0)+ds;
    Mat texture = this->getTetureInShape(this->images[0], newShape);
    Mat dt=texture-this->textureSet.row(0);
    double shapeChange=this->euqlidesNorm(ds);
    double textureChange=this->euqlidesNorm(dt);
    double textureOnShape=textureChange/shapeChange;
    cout<<"w: "<<textureOnShape<<endl;

    ds=this->randomPointsMat(10, this->b_g.cols);
    ds.convertTo(ds, b_g.type());
    newShape= this->b_g.row(0)+ds;
    Mat backProject=textureVariationPCA.backProject(newShape);
    backProject.convertTo(backProject, meanTexture.type());
    texture = meanTexture + backProject;
    texture.convertTo(texture, CV_8U);
    Mat realTexture;
    this->textureSet.row(0).convertTo(realTexture, CV_8U);
    Mat dG=texture-realTexture;
    shapeChange=this->euqlidesNorm(ds);
    textureChange=this->euqlidesNorm(dG);
    double bOnShape=textureChange/shapeChange;
    cout<<"w: "<<bOnShape<<endl;
    this->w=textureOnShape/bOnShape;
    cout<<"w: "<<w<<endl;

}

void AAM::makeApearanceModel()
{
    for(int i=0; i<deltaS.rows; i++)
    {
        Mat shapeParams=deltaS.row(i)*this->w;
        Mat textureParams=b_g.row(i);
        Mat appearanceRow;
        shapeParams.convertTo(shapeParams, textureParams.type());
        //cout<<"shape: "<<shapeParams.type()<<"  "<<shapeParams.rows<<"  "<<shapeParams.dims<<endl;
        //cout<<"texture: "<<textureParams.type()<<"  "<<textureParams.rows<<"  "<<textureParams.dims<<endl;
        hconcat(shapeParams, textureParams, appearanceRow);
        this->appearance.push_back(appearanceRow);
    }
    cout<<"appearance: "<<this->appearance.row(0)<<endl;
    this->appearancePCA(appearance, Mat(), CV_PCA_DATA_AS_ROW);
    this->cSet=appearancePCA.project(appearance);
    cout<<"cSet: "<<cSet.row(0)<<endl;
}

Mat AAM::sampleMat(Mat mat, int x)
{
    Mat result;
    Mat converted;
    mat.convertTo(converted, CV_32F);
    for(int i=0; i<mat.cols/x; i++)
    {

        result.push_back(converted.at<float>(0, i*x));
    }
    return result.t();
}

Mat AAM::getShapeParamsFromC(Mat c)
{
    Mat ap = this->appearancePCA.backProject(c);
    CvMat apMat=ap;
    CvMat shapeParams;
    cvGetCols(&apMat, &shapeParams, 0, shapeSet.cols);
    Mat shape=cvarrToMat(&shapeParams);
    shape=shape/this->w;
    return shape;
}

Mat AAM::getTextureParamsFromC(Mat c)
{
    Mat ap = this->appearancePCA.backProject(c);
    CvMat apMat=ap;
    CvMat textureParams;
    cvGetCols(&apMat, &textureParams, shapeSet.cols, shapeSet.cols+this->b_g.cols);
    Mat texture=cvarrToMat(&textureParams);
    return texture;
}

Mat AAM::getTetureInShape(Mat image, Mat shape)
{
    //cout<<"shape: "<<shape<<endl;
    Mat warpedImage;
    warpedImage=this->warpImage(image, shape);
    Mat texture=this->getTextureInsideHull(warpedImage, meanConvexHull);
    //cout<<"before normalizing and sample: "<<texture<<endl;
    Mat textureSampled=this->sampleMat(texture, sampling);
    //cout<<"sampled: "<<textureSampled<<endl;
    normalize(textureSampled, textureSampled, 0, 255, NORM_MINMAX);
    //cout<<"get texture in shape: "<<textureSampled<<endl;
    textureSampled.convertTo(textureSampled, CV_8U);
    return textureSampled;
}

Mat AAM::cutToRange(Mat mat, int min, int max)
{
    mat.convertTo(mat, CV_32S);
    for(int i=0; i<mat.cols; i++)
    {
        for(int j=0; j<mat.rows; j++)
        {
            if(mat.at<int>(j, i)<min)
            {
                mat.at<int>(j, i)=min;
            }
            if(mat.at<int>(j, i)>max)
            {
                mat.at<int>(j, i)=max;
            }
        }
    }
    return mat;
}


void AAM::saveModel(QString dir)
{
    this->saveMat(dir + "/A", this->A);
    this->saveMat(dir + "/shapes", this->shapeSet);
    this->saveMat(dir + "/textures", this->textureSet);
    this->saveMat(dir + "/appearance", this->appearance);
    QFile file(dir + "/parameters");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
    QTextStream out(&file);
    out<<this->ImageHeight<<endl<<ImageWidth<<endl<<sampling<<endl<<w<<endl<<E<<endl<<iterations;
    file.close();
    cout<<"Model saved"<<endl;
}

void AAM::loadModel(QString dir)
{
    this->A=this->loadMat(dir + "/A");
    this->shapeSet=this->loadMat(dir + "/shapes");
    cout<<"shape loaded"<<endl;
    this->makeShapeModel();
    triangulateMeanPoints();
    cout<<"trianguleate mean points"<<endl;
    this->meanConvexHull = this->createConvexHull(this->meanPoints);
    this->textureSet=this->loadMat(dir + "/textures");
    this->makeTextureModel();
    this->appearance=this->loadMat(dir + "/appearance");
    this->appearancePCA(appearance, Mat(), CV_PCA_DATA_AS_ROW);
    this->cSet=appearancePCA.project(appearance);
    this->displayModel();
    QFile file(dir + "/parameters");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;
    QTextStream in(&file);
    in>>this->ImageHeight>>ImageWidth>>sampling>>w>>E>>iterations;
    file.close();
    cout<<"Model loaded"<<endl;
}
