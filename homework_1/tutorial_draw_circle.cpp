#include <ros/ros.h>
#include <turtlesim/Pose.h>
#include <turtlesim/SpawnCircle.h>

int main(int argc, char** argv) {

  ros::init(argc, argv, "tutorial_draw_circle_node");
  ros::NodeHandle nh;
  
  ros::ServiceClient servizio=nh.serviceClient<turtlesim::SpawnCircle>("spawnCircle");
  int n=5;
  
  turtlesim::SpawnCircle::Response res;
  for(int i=0;i<n;i++){
	  float x=rand()/((float)RAND_MAX)*10;
	  float y=rand()/((float)RAND_MAX)*10;
	  turtlesim::SpawnCircle::Request req;
	  req.x=x;
	  req.y=y;
	  servizio.call(req,res);
	  //std::cerr << "fatto" << std::endl;
  }
}
