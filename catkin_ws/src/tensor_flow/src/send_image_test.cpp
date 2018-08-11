#include <iostream>
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/tracking.hpp>

#include "ros/ros.h"
//#include <ros/package.h>
#include <std_msgs/Bool.h>
#include "tensor_flow/image_srv.h"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>


int main(int argc, char **argv){
    
    ros::init(argc, argv, "send_image");

    ros::NodeHandle n;
   // ros::ServiceClient client = n.serviceClient<std_msgs::Bool>("/image_test");

    //cv::Mat imaRGB;

    return 0;
}

