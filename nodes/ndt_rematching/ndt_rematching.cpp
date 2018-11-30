/*
 * Copyright (c) 2018, ZhenRobotics, Inc.
 * All rights reserved.
 */

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/NavSatFix.h>
#include <autoware_msgs/ndt_stat.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

ros::Publisher start_pub;
ros::Publisher initialpose_pub;

static bool indoor = false;
static float linear_vel_x = 0.0;
static float linear_vel_y = 0.0;
static float angular_vel = 0.0;
static float fitness_score = 99999.99;
std_msgs::Int32 start_flag;
geometry_msgs::PoseStamped odom_pose;
geometry_msgs::PoseStamped gps_pose;
geometry_msgs::PoseStamped current_pose;
geometry_msgs::PoseStamped previous_pose;
static int arrived = 0;
static int out_control_counter = 0;
static bool in_control = true;

#define MAX_LINEAR_X 0.9
#define MAX_LINEAR_Y 0.1
#define MAX_ANGULAR 0.8


bool rematch_or_not()
{
    static int steady_counter = 0;
    if((abs(linear_vel_x) > MAX_LINEAR_X) || (abs(angular_vel) > MAX_ANGULAR) || (abs(linear_vel_y) > MAX_LINEAR_Y))
    {
        start_flag.data = 4;     // which is stop;
        start_pub.publish(start_flag);
        steady_counter = 0;
	ROS_INFO("FLYING");
        return true;
    }
    else 
    {
        if(fitness_score > 2)
        {
            start_flag.data = 4;     // which is stop;
            start_pub.publish(start_flag);
            steady_counter = 0;
	    ROS_INFO("FLYING");
            return true;
        }
        else
        {
            if(steady_counter++ > 30)
            {
                if(start_flag.data == 4)
                {
                    start_flag.data = 5;
                    start_pub.publish(start_flag);
		    ROS_INFO("confirm");
                }
		ROS_INFO("STEADY");
                return false;
            }
        }
    }
}

void rematching(geometry_msgs::PoseStamped msg)
{
    ros::Rate r(0.2);
    // stop the robot first;***********
    start_flag.data = 4;
    start_pub.publish(start_flag);
    static geometry_msgs::PoseWithCovarianceStamped initial_pose;
    //*********************************
    initial_pose.header = msg.header;
    initial_pose.pose.pose = msg.pose;
    boost::array<double, 36> covariance = {{
      0.25, 0, 0, 0, 0, 0,
      0, 0.25, 0, 0, 0, 0,
      0, 0, 1e6, 0, 0, 0,
      0, 0, 0, 1e6, 0, 0,
      0, 0, 0, 0, 1e6, 0,
      0, 0, 0, 0, 0, 0.06853891945200942
    }};
    initial_pose.pose.covariance = covariance;
    // then, publish the /initialpose with gps pose;
    //*********************************
    //initial_pose = construct_initialpose();
    initialpose_pub.publish(initial_pose);
    // in the end, start the robot again;**********
    r.sleep();
    start_flag.data = 5;
    start_pub.publish(start_flag);
}

void current_pose_callback(const geometry_msgs::PoseStampedPtr& _pose)
{
    current_pose.header.frame_id = "/map";
    current_pose.header.stamp = ros::Time::now();

    current_pose.pose = _pose -> pose;
}

void odom_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
{
    static geometry_msgs::PoseWithCovarianceStamped current_odom;
    static geometry_msgs::PoseWithCovarianceStamped pre_odom;
    static float offset_x = 0.0;
    static float offset_y = 0.0;
    static float offset_z = 0.0;
    current_odom.header.frame_id = "/map";
    current_odom.header.stamp = ros::Time::now();
    
    current_odom.pose = msg -> pose;

    if(arrived == 1)
    {
        static geometry_msgs::PoseStamped initial_odom;
        initial_odom.pose = current_odom.pose.pose;
        offset_x = current_odom.pose.pose.position.x - pre_odom.pose.pose.position.x;
        offset_y = current_odom.pose.pose.position.y - pre_odom.pose.pose.position.y;
        offset_z = current_odom.pose.pose.position.z - pre_odom.pose.pose.position.z;

        geometry_msgs::PoseWithCovarianceStamped predict_pose;
        predict_pose.header = current_odom.header;
        predict_pose.pose.pose.position.x = current_pose.pose.position.x + offset_x;
        predict_pose.pose.pose.position.y = current_pose.pose.position.y + offset_y;
        predict_pose.pose.pose.position.z = current_pose.pose.position.z + offset_z;
        predict_pose.pose.pose.orientation = previous_pose.pose.orientation;
        predict_pose.pose.covariance = current_odom.pose.covariance;
    }

}

void arrived_callback(const std_msgs::Int32ConstPtr& _arrive)
{
    arrived = _arrive -> data;
}

void cmd_vel_callback(const geometry_msgs::TwistConstPtr& _cmd)
{
    in_control = true;
    out_control_counter = 0;
}

//void gnss_callback(const sensor_msgs::NavSatFixPtr& fix)
//{
    // Or should use /gnss_pose, which is geometry_msgs::PoseStamped;
    // The problem is, we have no tranform between longitude/latitude and map frame;
//}

void gnss_callback(const geometry_msgs::PoseStampedConstPtr& msg)
{
    bool rematch_by_gps;
    gps_pose.header.frame_id = "/map";
    gps_pose.header.stamp = ros::Time::now();

    gps_pose.pose = msg -> pose;

    rematch_by_gps = rematch_or_not();
    if(rematch_by_gps){
	rematching(gps_pose);
    }

}

void ndt_stat_callback(const autoware_msgs::ndt_statConstPtr& _stat)
{
    fitness_score = _stat -> score;
}

void ndt_reliability_callback(const std_msgs::Float32ConstPtr& _reli)
{

}

void estimated_twist_callback(const geometry_msgs::TwistStampedConstPtr& _twist)
{
    linear_vel_x = _twist -> twist.linear.x;  // linear velocity in m/s;
    linear_vel_y = _twist -> twist.linear.y;  // linear velocity in m/s;
    angular_vel = _twist -> twist.angular.z;

    if(out_control_counter > 20)
    {
        in_control = false;
        rematch_or_not();
    }
    
    out_control_counter++;

    // if(rematch_or_not())
    // {
    //     rematching();
    // }
    // else
    // {
    //     if(start_flag.data == 4)
    //     {
    //         start_flag.data = 5;
    //         start_pub.publish(start_flag);
    //     }
    // }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "ndt_rematching");

    ros::NodeHandle n;
    ros::NodeHandle private_n("~");

    ros::Subscriber odom_sub = n.subscribe("robot_pose_ekf/odom_combined", 1, odom_callback);
    ros::Subscriber reli_sub = n.subscribe("/arrived", 1, arrived_callback);
    ros::Subscriber cmd_sub = n.subscribe("/cmd_vel", 1, cmd_vel_callback);
    ros::Subscriber stat_sub = n.subscribe("/ndt_stat", 1, ndt_stat_callback);
    ros::Subscriber twist_sub = n.subscribe("/estimate_twist", 1, estimated_twist_callback);
    ros::Subscriber gnss_sub = n.subscribe("/gnss_pose", 1, gnss_callback);
    // ros::Subscriber reli_sub = n.subscribe("/ndt_reliability", 1, ndt_reliability_callback);

    start_pub = n.advertise<std_msgs::Int32>("/start_flag", 1);  
    initialpose_pub = n.advertise<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1);  


    ros::spin();

    return 0;
}
