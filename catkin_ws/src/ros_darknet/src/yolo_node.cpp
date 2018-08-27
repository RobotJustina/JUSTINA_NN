#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <map>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "ros/ros.h"
#include <ros/package.h>
/*#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/Empty.h"
#include "std_msgs/Int32.h"
#include "std_msgs/String.h"
#include "justina_tools/JustinaTools.h"
#include "geometry_msgs/Point.h"
#include "tf/transform_listener.h"*/
#include <cv_bridge/cv_bridge.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "demo.hpp"

/*#include "vision_msgs/Cube.h"
#include "vision_msgs/CubesSegmented.h"
#include "vision_msgs/GetCubes.h"
#include "vision_msgs/FindPlane.h"
#include "vision_msgs/DetectObjects.h"

#include "visualization_msgs/MarkerArray.h"*/

//ros::NodeHandle* node;


int main(int argc, char** argv)
{
	
    std::cout << "Initializing yolo node by Hugo..." << std::endl;
    ros::init(argc, argv, "yolo_node");
    //ros::NodeHandle n;
    //node = &n;
    ros::NodeHandle nodeHandle("~");
    darknet_ros::YoloObjectDetector yoloObjectDetector(nodeHandle);	
    
    /*srvCubesSeg = n.advertiseService("/vision/cubes_segmentation/cubes_seg", callback_srvCubeSeg);
    srvCutlerySeg = n.advertiseService("/vision/cubes_segmentation/cutlery_seg", callback_srvCutlerySeg);
    cltRgbdRobot = n.serviceClient<point_cloud_manager::GetRgbd>("/hardware/point_cloud_man/get_rgbd_wrt_robot");
    cltFindPlane = n.serviceClient<vision_msgs::FindPlane>("/vision/geometry_finder/findPlane");
    cltExtObj = n.serviceClient<vision_msgs::DetectObjects>("/vision/obj_reco/ext_objects_above_planes");
    cltExtCut = n.serviceClient<vision_msgs::DetectObjects>("/vision/obj_reco/ext_objects_with_planes");
    ros::Subscriber subStartCalib = n.subscribe("/vision/cubes_segmentation/start_calib", 1, callbackStartCalibrate);
    ros::Subscriber subCalibV2 = n.subscribe("/vision/cubes_segmentation/calibv2", 1, callbackCalibrateV3);
    ros::Subscriber subCalibCutlery = n.subscribe("/vision/cubes_segmentation/calibCutlery", 1, callbackCalibrateCutlery2);
    ros::Publisher pubCubesMarker = n.advertise<visualization_msgs::MarkerArray>("/vision/cubes_segmentation/cubes_markers", 1);*/

    ros::Rate loop(30);
    
    std::cout << "Yolo.->Running..." << std::endl;
    
    //transformListener = new tf::TransformListener();
    
    while(ros::ok() && cv::waitKey(1) != 'q')
    {
        ros::spinOnce();
        loop.sleep();
    }
    
    //delete transformListener;

    cv::destroyAllWindows();   
}
