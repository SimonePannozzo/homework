#include <ros/ros.h>
#include <turtlesim/Pose.h>
#include <turtlesim/Color.h>
#include <turtlesim/GetCircles.h>
#include <turtlesim/RemoveCircle.h>

	
ros::ServiceClient  remove_circles;
ros::ServiceClient get_circles;
turtlesim::ColorConstPtr color;
turtlesim::PoseConstPtr pose;
std::vector<turtlesim::Circle> circles;
int colore;


float abspow2(float x){
	return abs(x*x);
}
void poseCallback(const turtlesim::PoseConstPtr& pose) {
	//check if pose is close to a circle and if the color is red
	//then remove it
	turtlesim::RemoveCircle::Request req;
	turtlesim::RemoveCircle::Response res;
	for(int i=0;i<circles.size();i++){
		float distanza=sqrt(abspow2(pose->x-circles[i].x)+(abspow2(pose->y-circles[i].y)));
		if(distanza<0.2 && colore==255){
			req.id=circles[i].id;
			remove_circles.call(req,res);
			circles=res.circles;
		}
	}
		
	
}


void colorCallback(const turtlesim::ColorConstPtr& color) {
  //save the current color seen by the turtle
  colore=color->r;
}

int main(int argc, char** argv) {

	ros::init(argc, argv, "tutorial_delete_circle_node");
	ros::NodeHandle nh;

	//run service clients for getting and removing circles
	get_circles=nh.serviceClient<turtlesim::GetCircles>("/getCircles"); 
	remove_circles=nh.serviceClient<turtlesim::RemoveCircle>("/removeCircle"); 
	
	turtlesim::GetCircles::Response res;			
	turtlesim::GetCircles::Request req;		
	get_circles.call(req,res);
	circles=res.circles;
	
	std::cerr << "e dai cane" << std::endl;
	
	//subscribe to the topics needed
	ros::Subscriber pose_sub=nh.subscribe("turtle1/pose",1,poseCallback);
	ros::Subscriber color_sub=nh.subscribe("turtle1/color_sensor",1,colorCallback);

	ros::spin();              //loop over the callbacks
}
