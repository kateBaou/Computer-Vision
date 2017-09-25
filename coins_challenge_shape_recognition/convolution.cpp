#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace cv;

void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);
void sobel(cv::Mat &input,cv::Mat &finalOutput,cv::Mat &finalOutputX,cv::Mat &finalOutputY);


//finding the xGradient
int xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y, x-1) +
                 image.at<uchar>(y+1, x-1) -
                  image.at<uchar>(y-1, x+1) -
                   2*image.at<uchar>(y, x+1) -
                    image.at<uchar>(y+1, x+1);
}

//finding the yGradient
int yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y-1, x) +
                 image.at<uchar>(y-1, x+1) -
                  image.at<uchar>(y+1, x-1) -
                   2*image.at<uchar>(y+1, x) -
                    image.at<uchar>(y+1, x+1);
}

void inc_if_inside(double *** H, int x, int y, int height, int width, int r )
{
  if (x>0 && x<width && y> 0 && y<height)
    H[y][x][r]++;
}

void hough(Mat &img_data, Mat &dist, double threshold, int minRadius, int maxRadius, double distance, Mat &h_acc, Mat &coins){
  int radiusRange = maxRadius - minRadius;
  int HEIGHT = img_data.rows;
  int WIDTH = img_data.cols;
  int DEPTH = radiusRange;

  double ***H;

  // Allocate memory
  H = new double**[HEIGHT];
  for (int i = 0; i < HEIGHT; ++i) {
    H[i] = new double*[WIDTH];

    for (int j = 0; j < WIDTH; ++j)
      H[i][j] = new double[DEPTH];
  }

    for(int y=0;y<img_data.rows;y++)  
  {  
       for(int x=0;x<img_data.cols;x++)  
       {  
        // printf("data point : %f\n", img_data.at<float>(y,x));
            if( (float) img_data.at<float>(y,x) > 250.0 )  //threshold image  
            {   
              for (int r=minRadius; r<radiusRange; r++)
              {

                int x0 = round(x + r * cos(dist.at<float>(y,x)) );
                int x1 = round(x - r * cos(dist.at<float>(y,x)) );
                int y0 = round(y + r * sin(dist.at<float>(y,x)) );
                int y1 = round(y - r * sin(dist.at<float>(y,x)) );


                inc_if_inside(H,x0,y0,HEIGHT, WIDTH, r);
                 // inc_if_inside(H,x0,y1,HEIGHT, WIDTH, r);
                 // inc_if_inside(H,x1,y0,HEIGHT, WIDTH, r);
                inc_if_inside(H,x1,y1,HEIGHT, WIDTH, r);
              }
            }  
       }  
  }  

  //create 2D image by summing values of the radius dimension
  for(int y0 = 0; y0 < HEIGHT; y0++) {
    for(int x0 = 0; x0 < WIDTH; x0++) {
      for(int r = minRadius; r < radiusRange; r++) {
        h_acc.at<float>(y0,x0) +=  H[y0][x0][r];// > 1 ? 255 : 0;
       // printf("h : %d", H[y0][x0][r]);
      }
    }
  }

  std::vector<Point3f> bestCircles;
  
  //compute optimal circles
  for(int y0 = 0; y0 < HEIGHT; y0++) {
    for(int x0 = 0; x0 < WIDTH; x0++) {
      for(int r = minRadius; r < radiusRange; r++) { 
        if(H[y0][x0][r] > threshold){
          Point3f circle(x0, y0, r);
          int i;
          for(i = 0; i < bestCircles.size(); i++) {
            int xCoord = bestCircles[i].x;
            int yCoord = bestCircles[i].y;
            int radius = bestCircles[i].z;            
            if(abs(xCoord - x0) < distance && abs(yCoord - y0) < distance) {           
              if(H[y0][x0][r] > H[yCoord][xCoord][radius]) {
                bestCircles.erase(bestCircles.begin()+i);
                bestCircles.insert(bestCircles.begin(), circle);
              }
              break;
            }
          }
          if(i == bestCircles.size()){
            bestCircles.insert(bestCircles.begin(), circle);
          }
        }
      }
    }
  }

  for(int i = 0; i < bestCircles.size(); i++) {
    int lineThickness = 4;
    int lineType = 10;
    int shift = 0;
    int xCoord = bestCircles[i].x;
    int yCoord = bestCircles[i].y;
    int radius = bestCircles[i].z;
    Point2f center(xCoord, yCoord);      
    circle(coins, center, radius-1, Scalar(255,0,0), lineThickness, lineType, shift);
  }
}



int main( int argc, char** argv )
{

 Mat image,h_acc;
Mat gray_image;
Mat coinsBlurred;
 image = imread( "coins1.jpg", 1 );

 
 
// CONVERT COLOUR, BLUR AND SAVE
 
 cvtColor( image, gray_image , CV_BGR2GRAY );
  GaussianBlur(gray_image ,23,coinsBlurred);
 
 Mat imgFinal;
 Mat imgFinalx;
 Mat imgFinaly;
 imgFinal=coinsBlurred.clone();
 imgFinalx=coinsBlurred.clone();
 imgFinaly=coinsBlurred.clone();

    for(int x = 0; x < coinsBlurred.rows; x++)
        {    
		for(int y = 0; y < coinsBlurred.cols; y++)
         	{
		       imgFinal.at<uchar>(x,y) = 0.0;
                       imgFinalx.at<uchar>(x,y) = 0.0;
		       imgFinaly.at<uchar>(x,y) = 0.0;
		}
	}

 sobel(coinsBlurred,imgFinal, imgFinalx, imgFinaly);
 normalize(imgFinalx,imgFinalx, 0, 255, NORM_MINMAX, -1, Mat());
 imwrite( "sobelx.jpg", imgFinalx );
 imwrite( "sobely.jpg", imgFinaly );
 imwrite( "sobel.jpg", imgFinal );
 h_acc.create(imgFinal.rows, imgFinal.cols, CV_32FC1);
 hough(imgFinal,imgFinalx,10, 15, 100, 45, h_acc, image);
 imwrite("hough.jpg",image);
 imwrite("houghSpace.jpg",h_acc);
 return 0;
}


void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);
	
	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}



void sobel(cv::Mat &input,cv::Mat &finalOutput,cv::Mat &finalOutputX,cv::Mat &finalOutputY)
{  

 	int gx, gy, sum;
        double angle;
    for(int y = 0; y < input.rows; y++)
 	{ 
          for(int x = 0; x < input.cols; x++)
         	{      
                        //calculate the xGradient
			gx=xGradient(input, x , y);
                        //calculate the yGradient
			gy=yGradient(input,x,y);
			//find a better way to fix the final
                        sum = sqrt(gx*gx+gy*gy)> 100 ? 255 : 0;
			angle=atan2f(gx,gy);
                        finalOutput.at<uchar>(y,x) = sum;
			finalOutputX.at<uchar>(y,x)=angle;
			finalOutputY.at<uchar>(y,x)=abs(gy);
                        	
		}
	}


}
