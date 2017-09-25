
// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

void mergeOverlappingBoxes(std::vector<cv::Rect> &inputBoxes, cv::Mat &image, std::vector<cv::Rect> &outputBoxes)
{
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Mask of original image
    cv::Size scaleFactor(5,5); // To expand rectangles, i.e. increase sensitivity to nearby rectangles. Doesn't have to be (10,10)--can be anything
    for (int i = 0; i < inputBoxes.size(); i++)
    {
        cv::Rect box = inputBoxes.at(i) + scaleFactor;
        cv::rectangle(mask, box, cv::Scalar(255), CV_FILLED); // Draw filled bounding boxes on mask
    }

    std::vector<std::vector<cv::Point> > contours;
    // Find contours in mask
    // If bounding boxes overlap, they will be joined by this function call
    cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int j = 0; j < contours.size(); j++)
    {
        outputBoxes.push_back(cv::boundingRect(contours.at(j)));
    }
}




void sobel(Mat img, Mat &dx, Mat &dy, Mat &mag, Mat &dist)
{
  float acc_dx = 0, acc_dy = 0;         
  float k1 [] = {-1,-2,-1,0,0,0,1,2,1}; 
  float k2 [] = {-1,0,1,-2,0,2,-1,0,1};

  for(int i=0; i<img.rows; i++) {
    for(int j=0; j<img.cols; j++) {
      acc_dx = acc_dy = 0;

      //apply kernel/mask
      for (int nn=-1; nn<2; nn++) {
        for (int mm = -1; mm < 2; mm++) {
          //if (i + nn > 0 && i + nn < img.rows && j + mm > 0 && j + mm < img.cols) {
            acc_dx += (float)img.at<uchar>(i+nn,j+mm) * k1[((mm+1)*3) + nn + 1];
            acc_dy += (float)img.at<uchar>(i+nn,j+mm) * k2[((mm+1)*3) + nn + 1];
          //}
        }
      }
      //write final values
      dx.at<float>(i,j) = acc_dx;
      dy.at<float>(i,j) = acc_dy;
      mag.at<float>(i,j) = (sqrtf(acc_dy*acc_dy + acc_dx*acc_dx)) > 100 ? 255 : 0;
      dist.at<float>(i,j) = atan2f(acc_dy,acc_dx);
      
    }
  }
}

void inc_if_inside(double *** H, int x, int y, int height, int width, int r )
{
  if (x>0 && x<width && y> 0 && y<height)
    H[y][x][r]++;
}
//Hough Circle detection function.Returns the final image and the hough space
void hough(Mat &grad, Mat &dir, double threshold, int minRadius, int maxRadius, double distance, Mat &hSp, Mat &frame,vector<Point3f>& bestCircles)
{
  int radiusRange = maxRadius - minRadius;
  int HEIGHT = grad.rows;
  int WIDTH = grad.cols;
  int DEPTH = radiusRange;

  double ***H;

  // Allocate memory
  H = new double**[HEIGHT];
  for (int i = 0; i < HEIGHT; ++i) {
    H[i] = new double*[WIDTH];

    for (int j = 0; j < WIDTH; ++j)
      H[i][j] = new double[DEPTH];
  }

    for(int y=0;y<grad.rows;y++)  
  {  
       for(int x=0;x<grad.cols;x++)  
       {  
        
            if( (float) grad.at<float>(y,x) > 250.0 )  //threshold image  
            {   
              for (int r=minRadius; r<radiusRange; r++)
              {

                int x0 = round(x + r * cos(dir.at<float>(y,x)) );
                int x1 = round(x - r * cos(dir.at<float>(y,x)) );
                int y0 = round(y + r * sin(dir.at<float>(y,x)) );
                int y1 = round(y - r * sin(dir.at<float>(y,x)) );
		
		if (x0>0 && x0<grad.cols && y0> 0 && y0<grad.rows)
		{
    			H[y0][x0][r]++;
		}
		if (x1>0 && x1<grad.cols && y1> 0 && y1<grad.rows)
		{
    			H[y1][x1][r]++;
		}
                
              }
            }  
       }  
  }  

  //create 2D representation by summing values of the radius dimension
  for(int y0 = 0; y0 < HEIGHT; y0++) {
    for(int x0 = 0; x0 < WIDTH; x0++) {
      for(int r = minRadius; r < radiusRange; r++) {
        hSp.at<float>(y0,x0) +=  H[y0][x0][r];
       
      }
    }
  }

  
  
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

  
}


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect darts and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}



/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
        Mat img_1 = imread("dart.bmp", CV_LOAD_IMAGE_GRAYSCALE );
	
	std::vector<Rect> darts;
	std::vector<Rect> unionBoxes;
	Mat frame_gray,frame_blur;
  	
	//Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
        GaussianBlur( frame_gray , frame_blur , Size(9, 9), 2, 2 );
////////
int minHessian = 15000;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( frame_gray, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;
  std::vector<Point2f> d_surf;
  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( frame_gray, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors with a brute force matcher
  BFMatcher matcher(NORM_L2);
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );




 for( int i = 0; i < matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    d_surf.push_back( keypoints_1[ matches[i].queryIdx ].pt );
  }

  //-- Show detected matches
  //imwrite("Matches.jpg", img_matches );
//////


	
	//Initialize Variables needed for the sobel function
	Mat grad_x;
	Mat grad_y;
	Mat gradient;
	Mat direction;

	grad_x.create(frame_blur.rows, frame_blur.cols, CV_32FC1);
	grad_y.create(frame_blur.rows, frame_blur.cols, CV_32FC1);
	gradient.create(frame_blur.rows, frame_blur.cols, CV_32FC1);
	direction.create(frame_blur.rows, frame_blur.cols, CV_32FC1);
	
	//Run the sobel function
	std::cout << "running the sobel function" << std::endl;
        sobel(frame_gray,grad_x,grad_y,gradient, direction);

     	normalize(grad_x, grad_x, 0, 255, NORM_MINMAX, -1, Mat());
  	normalize(grad_y, grad_y, 0, 255, NORM_MINMAX, -1, Mat());
  	//normalize(direction,direction, 0, 255, NORM_MINMAX, -1, Mat());

	imwrite( "gradient.jpg", gradient);
	
	//Detect circles
	//Define the matrix for the hough space
	Mat houghSpace;
	houghSpace.create(gradient.rows, gradient.cols, CV_32FC1);
	vector<Point3f> detectedCircles;
	cout << "detecting circles" << endl;
	hough(gradient, direction, 11, 15 ,100, 45,houghSpace,frame,detectedCircles); 


	imwrite( "hspace.jpg", houghSpace);
	std::cout << "number of circles detected by hough is" << std::endl;
	std::cout << detectedCircles.size() << std::endl;
	/*for(int i = 0; i < detectedCircles.size(); i++) 
	{      
    		int lineThickness = 4;
    		int lineType = 10;
    		int shift = 0;
    		int xCoord = detectedCircles[i].x;
    		int yCoord = detectedCircles[i].y;
    		int radius = detectedCircles[i].z;
    		Point2f center(xCoord, yCoord);      
    		circle(frame, center, radius-1, Scalar(0,255,0), lineThickness, lineType, shift);
  	}
	imwrite( "detectedCircles.jpg",frame);*/
	
	//Detect lines in the image using the in-built function
	vector<Vec4i> lines;
        Mat gradientLines=gradient.clone();
	gradientLines.convertTo(gradientLines, CV_8UC1);
	std::cout << "detecting lines using the probabilistic function" << std::endl;
	HoughLinesP(gradientLines,lines,1,CV_PI/180,150,60,10);

	//Draw the detected lines	
	
	for( size_t i = 0; i < lines.size(); i++ )
	{
  
  		Vec4i l = lines[i];
  		line( gradient , Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
	}
	
   	imwrite( "gradientlines.jpg", gradient);
	
	  
     

		
    //Perform Viola-Jones Object Detection and the hough transform for circles
	std::cout << "performing Viola-Jones detection" << std::endl;
     cascade.detectMultiScale( frame_gray, darts, 1.1, 1 , 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
     
   // Print number of darts found
     std::cout << "Number of darts found by Viola" << std::endl;
     std::cout << darts.size() << std::endl;	
  
     	/*for(int i=0;i<darts.size();i++)
	{ 
          std::cout << i << std::endl;
  		rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 0, 0 ), 2);
 	}*/   
	
	//Merge the boxes that were close to each other to form a bigger area
	mergeOverlappingBoxes(darts,frame, unionBoxes);  

	// Print number of darts found
	 std::cout << "Final number of overlapping darts" << std::endl;
	std::cout << unionBoxes.size() << std::endl;
	
	//Classify these boxes as interesting or not

	for(int i=0;i<unionBoxes.size();i++)
	{
		int unionXCoord=unionBoxes[i].x;
		int unionYCoord=unionBoxes[i].y;
		int unionWidth=unionBoxes[i].width;
		int unionHeight=unionBoxes[i].height;
		std::cout << "finding areas of interest" << std::endl;
		//FIND THE UNION BOX THAT CONTAINS THE CIRCLES AND THE LINES
			bool found=false;

			for(int k=0;k<lines.size();k++)
			{
				Vec4i l=lines[i];
				if(unionXCoord<=l[0] || unionYCoord<=l[1] || unionXCoord+unionWidth>=l[2] || unionYCoord+unionHeight>=l[3])
				{       
					for(int l=0;l<detectedCircles.size();l++)
					{	
						if(unionXCoord<=detectedCircles[l].x && unionYCoord<=detectedCircles[l].y || unionWidth>=detectedCircles[l].z)
						{
							found=true;
						}
					}
				}
			}
			
			if(found)
			{				
			//rectangle(frame, Point(unionBoxes[i].x, unionBoxes[i].y), Point(unionBoxes[i].x + unionBoxes[i].width, unionBoxes[i].y + unionBoxes[i].height), Scalar( 255 , 0 , 0 ), 2);
			
			std::cout << "find the dartboard that best fits the area of interest" << std::endl;
			//find the results produced by viola that are included inside this unionBox
			for(int j=0;j<darts.size();j++)
			{				
						
				int dartXCoord=darts[j].x;
				int dartYCoord=darts[j].y;
				int dartWidth=darts[j].width;
				int dartHeight=darts[j].height;
				bool foundDart=false;
				
				if(dartXCoord>=unionXCoord && dartXCoord+dartWidth<=unionXCoord+unionWidth && dartYCoord>=unionYCoord && dartYCoord+dartHeight<=unionYCoord+unionHeight)//check only the boxes included in the union box
				{
					
						for(int k=0;k<lines.size();k++)
						{
							Vec4i l=lines[i];
							if(dartXCoord<=l[0] || dartYCoord<=l[1] || dartXCoord+dartWidth>=l[2] || dartYCoord+unionHeight>=l[3])
							{       
							
							
							for(int l=0;l<detectedCircles.size();l++)
							{	
							//Centre inside the rectangles
							if(dartXCoord<=detectedCircles[l].x && dartYCoord<=detectedCircles[l].y && dartXCoord+dartWidth>=detectedCircles[l].x && dartYCoord+dartHeight>=detectedCircles[l].y)
							{
							foundDart=true;
							}//check on the left of the circle
							
							
						}
						}
						}
				}
						if(foundDart)
          {	
						rectangle(frame, Point(darts[j].x, darts[j].y), Point(darts[j].x + darts[j].width, darts[j].y + darts[j].height), Scalar( 0 , 255 , 0 ), 2);
          }
  //-- Draw matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, frame, keypoints_2, matches, img_matches );
		imwrite("Matches.jpg", img_matches );	
	   }
  }
}
			int TP,FP,FN;
        // Show current tally
        cout << "\nThe number of true positives (TP) are: TP = \n";
	cin>> TP ;
        cout << "The number of false positives (FP) are: FP = \n";
	cin>> FP;
        cout << "The number of false negatives (FN) are: FN = \n";
	cin>> FN;
        

    
    
     // Now calculate the precision recall based on the TP FP FN
        double precision = (double)TP / (double)(TP + FP);
        double recall = (double)TP / (double)(TP + FN);
        
        double f1_score = 2.0 * (precision * recall) / (precision + recall); 
        
     cout << "Precision = " << precision<< "." << endl;
     cout << "Recall = " << recall << "." << endl;
     cout << "F1 Score = " << f1_score << "." << endl;		
}
		
		
		
	

		
  




