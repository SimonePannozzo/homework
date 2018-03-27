#include <iostream>
#include <string>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

namespace filesystem = boost::filesystem;
using namespace std;

// Compute the euclidean distance between two points
float pointsDist(const cv::Point2f &p1, const cv::Point2f &p2 )
{
  return std::sqrt( (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) );
}

/* Increses the accumulator by one unit on points belonging to 
 * a specific circumference defined by center and radius.
 * The template point define the cv::Mat accumulator type (e.g., int, float, ...)
 */
template <typename _T> void accumulateCircle( cv::Mat &accumulator, const cv::Point &center, int radius )
{
  int x = -radius, y = 0, err = 2-2*radius, px, py;
  do
  {
    py = center.y+y; px = center.x-x; /* I. Quadrant */
    if ( ( unsigned ) px < ( unsigned ) ( accumulator.cols ) && ( unsigned ) py < ( unsigned ) ( accumulator.rows ) )
      accumulator.at<_T>(py, px)++; 
    
    py = center.y-x; px =  center.x-y; /* II. Quadrant */
    if ( ( unsigned ) px < ( unsigned ) ( accumulator.cols ) && ( unsigned ) py < ( unsigned ) ( accumulator.rows ) )
      accumulator.at<_T>(py, px)++; 
    
    py = center.y-y; px = center.x+x; /* III. Quadrant */
    if ( ( unsigned ) px < ( unsigned ) ( accumulator.cols ) && ( unsigned ) py < ( unsigned ) ( accumulator.rows ) )
      accumulator.at<_T>(py, px)++; 
    
    py = center.y+x; px = center.x+y; /* IV. Quadrant */
    if ( ( unsigned ) px < ( unsigned ) ( accumulator.cols ) && ( unsigned ) py < ( unsigned ) ( accumulator.rows ) )
      accumulator.at<_T>(py, px)++; 
    
    radius = err;
    if (radius <= y) 
      err += ++y*2+1; /* e_xy+e_y < 0 */
    if (radius > x || err > y)
      err += ++x*2+1; /* e_xy+e_x > 0 or no 2nd y-step */
  } while (x < 0);
}

/* Compute the edge map of an input image using a very simple 
 * algorithm based on a thresholded gradient magnitude image (see slides).
 * You may also try the Canny algorithm implemented in openCV. */
void detectEdges( const cv::Mat &src_img, cv::Mat &edge_map )
{
  cv::Mat src_img_f;
  
  src_img.convertTo( src_img_f,cv::DataType<float>::type );
  // Map intensities in the range [0,1]
  cv::normalize(src_img_f, src_img_f, 0, 1.0, 
                cv::NORM_MINMAX, cv::DataType<float>::type);
  
  cv::Mat dx_img, dy_img;
  cv::Sobel(src_img_f, dx_img, cv::DataType<float>::type, 1, 0, 3);
  cv::Sobel(src_img_f, dy_img, cv::DataType<float>::type, 0, 1, 3);
  
  cv::Mat gradient_mag_img, abs_dx_img, abs_dy_img, binary_img;
  
  // Compute the gradient magnitude image
  abs_dx_img = cv::abs(dx_img);
  abs_dy_img = cv::abs(dy_img);
  gradient_mag_img = 0.5*(abs_dx_img + abs_dy_img);
      
  // Binarize the image
  cv::threshold ( gradient_mag_img, edge_map, 0.4, 1.0,cv::THRESH_BINARY );

  // Convert the floating point edge map in a unsigned char edge map (255: edge point, 0: no edge point)
  cv::normalize(edge_map, edge_map, 0, 255, 
                cv::NORM_MINMAX, cv::DataType<uchar>::type);
  
//   // Debug code: show the edge map
//   cv::imshow("Edge map",edge_map);
//   // Wait a key
//   cv::waitKey();
}

/* Detect the "strongest" circular object in the input image with radius
 * between min_radius and max_radius.
 * You should implement a (simplified) version of the Circle Hough Transform, i.e.:
 * 
 * - Compute an edge map of the input image (e.g., you may use the provided function detectEdges()
 *   or the openCV function cv::Canny())
 * - For each searched radius r (min_radius <= r <= max_radius)
 *      - Inizialize to zero an accumulator (e.g., a cv::Mat with int type) with the same size 
 *        of the input image
 *      - For each non-zero point (x,y) in the edge map, accumulate in the accumulator a circle with 
 *        radius r and center (x,y) (see the accumulateCircle() function defined above)
 *      - Extract from the accumulator the point with the highest value: the coordinates of this point
 *        represent the center of the strongest circular object with radius r
 * - Select the strengest object among all radius
 * - Provide in output the center and the radius of the strongest circular object 
 */ 
void findCircularShape( const cv::Mat src_img, cv::Point &center, int &radius, int min_radius = 50, int max_radius = 100){
  cv::Mat edge_map;
  detectEdges(src_img,edge_map);
  int r;
  int max=-1; 
  for(r=min_radius;r<=max_radius;r++){
	  cv::Mat accumulator=cv::Mat::zeros(src_img.size(),CV_32SC1);
	  for(int x=0;x<accumulator.rows;x++){
		  for(int y=0;y<accumulator.cols;y++){
			  if (edge_map.at< unsigned char >(x,y) > 0){
				accumulateCircle<int>(accumulator,cv::Point(x,y),r);
			  }
		  }
	  }
	  for(int x=0;x<accumulator.rows;x++){
		  for(int y=0;y<accumulator.cols;y++){
			 if(accumulator.at<int>(x,y)>max){
				 max=accumulator.at<int>(x,y);
				 center=cv::Point(x,y);
				 radius=r;
			 }
		  }
	  }
  }
}

// Draw into the image img a sequence of 2D points
void drawPoints( cv::Mat &img, const std::vector< cv::Point2f > &pts, cv::Scalar color )
{
  int pts_size = pts.size(), width = img.cols, height = img.rows;
  for(int i = 0; i< pts_size; i++)
  {
    int xc = cvRound(pts[i].x), yc = cvRound(pts[i].y);
    if ( unsigned(xc) < unsigned(width) && unsigned(yc) < unsigned(height) )
      cv::circle( img, cv::Point(xc,yc),3,color,-1,8 );
  }
}


int main(int argc, char **argv)
{
  if(argc < 2)
  {
    cout<<"Usage : "<<argv[0]<<" <images directory>"<<endl;
    return 0;
  }
  
  // Loads all the files in the directory argv[1]
  filesystem::path images_path (argv[1]);
  filesystem::directory_iterator end_itr;
  vector<string> paths;
  for (filesystem::directory_iterator itr(images_path); itr != end_itr; ++itr)
  {
    if ( !filesystem::is_regular_file(itr->path()))
      continue;
    paths.push_back(itr->path().string());
  }
  
  // Sort the files
  sort(paths.begin(), paths.end());
  
  bool circle_found = false;
  std::vector< cv::Point2f > features, tracks;
  cv::Mat prev_img;
  
  // Try to open all files as images
  for( auto &img_path: paths )
  {
    // Load the color images (use it to show the results)
    cv::Mat src_img = imread ( img_path, cv::IMREAD_COLOR ), src_img_gl;
    
    if( src_img.empty())
      // Not an image: continue
      continue;
    
    // Smooth the image to remove some noise
    cv::GaussianBlur( src_img, src_img, cv::Size(0,0), 1 );
    // Convert the color images to graylevels (used it for processing)
    cv::cvtColor(src_img, src_img_gl, cv::COLOR_BGR2GRAY) ;
    
    
    if( circle_found )
    {
      /*********** TRACKING ***********/
      vector<uchar> status;
      std::vector<float> errors;
      
      cv::Size tracking_win_size ( 7, 7);
      cv::calcOpticalFlowPyrLK ( prev_img, src_img_gl, features, tracks,
                                 status, errors, tracking_win_size, 3 );
      
          
      
      
      /* Now the current tracks will become the features points to be tracked for the next 
       * prev_img-src_img_gl pair: you could just set features=tracks (try it!).
       * 
       * In the next code lines, we here try to remove some outliers, i.e. some bad tracks: 
       * we compute the average (mean) optical flow dislacements, and discard all the tracks with 
       * displacement greater than 5X the current standard deviation... it is a rather basic outlier 
       * removal strategy, of course not the best one, but does its job */
      
      vector<float> flow_displacement;
      flow_displacement.reserve(features.size());
      double mean = 0, std_dev = 0;
      int num_valid_tracks = 0;
      // Compute the displacement mean
      for( int i = 0; i < int(features.size()); i++ )
      {
        float dist = 0;
        // The flow of the corresponding features has been found?
        if( status[i] )
        {
          dist = pointsDist(features[i], tracks[i]);
          mean += dist;
          num_valid_tracks++;
        }
        flow_displacement.push_back(dist);
      }
      if(!num_valid_tracks) num_valid_tracks++;
      mean /= num_valid_tracks;
      
      // Compute the displacement standard deviation
      for( int i = 0; i < int(flow_displacement.size()); i++ )
      {
        if( status[i] )
          std_dev += (flow_displacement[i] - mean)*(flow_displacement[i] - mean);
      }
      std_dev = sqrt(std_dev/num_valid_tracks);
      
      /* Select form the the current features only the ones that have deviation less than 
       * a multiple of the standard deviation, in our case 5X of standard deviation. */
      features.clear();
      for( int i = 0; i < int(flow_displacement.size()); i++ )
      {
        if( status[i] && fabs( flow_displacement[i] - mean ) < 5*std_dev )
          features.push_back(tracks[i]);
      }
      tracks.clear();
    }
    else
    {

      /*********** DETECTION ***********/
      // Only for the first image, detect a circular object
      
      
      // The actual circle center wil bre provided by findCircularShape()      
      cv::Point center(0,0);
      // The actual circle radius wil bre provided by findCircularShape()
      int radius = 10;
      findCircularShape(src_img_gl, center, radius);
      cv::circle(src_img, center, radius, cv::Scalar(0,0,255), 3);
      
      // Show the found circle
      cv::imshow("Input image",src_img);
      // Wait 1000 millisecnods
      cv::waitKey(1000);
    
      /*********** FEATURES EXTRACTION ***********/
      
      /* Prepare an unsigned char mask: 0 outside the circle, 255 inside, to be used to detect features
       * only inside the circular object */
      cv::Mat mask = cv::Mat::zeros( src_img_gl.size(), cv::DataType<uchar>::type );
      cv::circle(mask, center, radius - 2, cv::Scalar(255), -1);
      
      
	  //già dichiarati
	  //std::vector< cv::Point2f > features, tracks;     
	  
	  double quality_level = 0.2;
      int max_features = 100, min_distance = 7;
      cv::goodFeaturesToTrack ( src_img_gl, features, max_features,
                                quality_level, min_distance,mask);         
      
      
    
      
      
      /* We have detected the circular object and detected the related features:
       * for the remainig images, just track these features */
      circle_found = true;
    }
    
    // Draw the features at the current image
    drawPoints( src_img, features, cv::Scalar(0,0,255) );
    // Show the features
    cv::imshow("Input image",src_img);
    // Wait 10 millisecnods
    cv::waitKey(10);
    
    // Now the current image becomes the previous one
    prev_img = src_img_gl;
  }
  
  cout<<"Type ESC to exit"<<endl;
  while( cv::waitKey(10) != 27 );
  return 0;
}
