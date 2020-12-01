# Talking to the real Jaco

## Dependencies

- [Dockerfile](https://github.com/melfm/jaco_docker)
- [Ros Interface](https://github.com/johannah/ros_interface)

## Running
Run `roslaunch jaco_remote.launch` from inside the container. If `roslaunch` is not finding the launchfile, just run it from `ros_interface/launch`.

Now run `simple_robot.py` from outside of the container.

## Commands
### Building Docker
These instructions are for the docker with Nvidia support. Just don't forget to build with the same name and make sure the runner script is trying to launch the correct one!
```
docker build --tag jaco_control_nvidia .
```

```
 chmod a+x run_docker_image.bash
```

```
./run_docker_image.bash
```

Whenever you run out of space, just clear all the dockers and rebuild. At the moment, its building the images in the root directory and my current root is low on storage.
```
docker system prune -a
```


### Running
* To access the arm via USB copy the udev srule file `10-kinova-arm.rules` from `/udev` to
`/etc/udev/rules.d` and then connect the robot to your machine:
```
sudo cp udev/10-kinova-arm.rules /etc/udev/rules.d/
```

* Run the Docker container with access to the USB devices for establishing connection to the robot:
```
docker run -it --name jaco_robot_net --privileged -v /dev/bus/usb:/dev/bus/usb  -p 9030:9030 jaco_control
```

To connect to the running container:
```
docker exec -it jaco_robot_net /bin/bash
```

To create a backup for the container:
```
docker save -o jaco_robot.tar jaco_control
```

RealSense Camera Launcher
```
roslaunch realsense2_camera rs_camera.launch
```

## Running Rviz on docker
- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)
- Then try these steps [1.3 nvidia-docker2](http://wiki.ros.org/action/login/docker/Tutorials/Hardware%20Acceleration#nvidia-docker2)
If the original nvidia run script doesn't work, try [this](https://answers.ros.org/question/300113/docker-how-to-use-rviz-and-gazebo-from-a-container/)

Now you can run:
```
rosrun rviz rviz
```

When you launch rviz, to see the frames, make sure Fixed frame is set to `camera_color_frame` although oddly it shows up with an artifact.


### Troubleshooting
These instructions worked for me the very first time, and then stopped working for some reason. If you run into issues such as:
```
xauth:  /tmp/.docker.xauth not writable, changes will be ignored
xauth: (argv):1:  unable to read any entries from file "(stdin)"
chmod: changing permissions of '/tmp/.docker.xauth': Operation not permitted
```

It also told me to use `XAUTH=/tmp/.docker.xauth-n` instead of `XAUTH=/tmp/.docker.xauth` (maybe because I had ran it more than once and it had created multiple temp files). You may or may not need to run `xhost +si:localuser:root`. And finally, running the runner script with `sudo` worked.

## Debugging
### RealSense build issues
```
E: Failed to fetch http://packages.ros.org/ros/ubuntu/dists/bionic/InRelease  Clearsigned file isn't valid, got 'NOSPLIT' (does the network require authentication?)
E: The repository 'http://packages.ros.org/ros/ubuntu bionic InRelease' is not signed.
```
Solution
```
sudo vim /etc/resolv.conf
nameserver 8.8.8.8
```

### Port Blocking Issue
Sometimes the camera launch node wont find the camera, just unplug the UCB and plug back in.


```
Resource not found: realsense2_camera
ROS path [0]=/opt/ros/melodic/share/ros
ROS path [1]=/opt/ros/melodic/share
The traceback for the exception was written to the log file
```

Or

```
[ INFO] [1602801703.992486348]: Running with LibRealSense v2.39.0
[ WARN] [1602801703.999530191]: No RealSense devices were found!
```
Or

```
[ERROR] [1602802733.479629785]: KinovaCommException: Could not initialize Kinova API (return code: 1015)

process[j2s7s300_state_publisher-3]: started with pid [2700]
[j2s7s300_driver-2] process has died [pid 2699, exit code -11, cmd /root/catkin_ws/devel/lib/kinova_driver/kinova_arm_driver j2s7s300 __name:=j2s7s300_driver __log:=/root/.ros/log/fe87d17c-0f39-11eb-a8aa-04d9f5f4a2f1/j2s7s300_driver-2.log].
log file: /root/.ros/log/fe87d17c-0f39-11eb-a8aa-04d9f5f4a2f1/j2s7s300_driver-2*.log
```

Any of the errors above are pretty much related to USB ports being blocked. I'm not sure how this happens, but just switching to another port works. I am hoping there is a way to unblock a port when this happens.