# SRL Moonshot Members Exclusive

>**Note**: This is meant for SRL Moonshot members. If you have stumbled upon this and you are not a SRL Moonshot member, then I doubt you will find it very useful. 

This short guide will describe how you can get relevant USD files to start training even quicker. It will also mention the changes to the URDF files.

## How to get the already made USD files

In the shared drive you can find a ZIP file named `usd_files.zip` containing a directory called `usd/`. This ZIP file is located at 
```
SRL-Moonshot > 1. Meeting Slides & Materials > Marcus > robot_descriptions 
```
In here you will find a `robot` and a `terrain` directory that contain ready-to-use-already-tested USD files. With this you can get started even quicker. The folder contains USD files that were made by me. They were made based on an alterated URDF file, which was originally made by first compiling the xacro files for Hero Vehicle and Hero Dragon. You should place these folders in the directory `moonshot/descriptions/usd/`. 

## Creating new USD files but based on edited URDF

I have included these alterated URDF files in another ZIP file called `urdf_files.zip`. If you place these in the `urdf` folder in the `hero_ros2` package (see SRL GitHub), then you can create new USD files based on these if you would like. They need to be placed here so that they can correctly reference the meshes. To create the new USD files, you can follow the guide in the `README.md`. 

### *D-Did you just create ANOTHER URDF version??*

Yeah, I did what had to be done. I made several changes that facilitate the RL process and import into Isaac Lab. The changes that were made to the URDF can be found here:

#### URDF changes for Vehicle

- leg1joint[3,5] changed from “continuous” to “fixed”
- leg1joint[2,4,6] changed from “continuous” to “revolute”
- leg1grip[1,2], leg1grip[1,2]bis changed from “prismatic” to “fixed”
- leg1joint4 origin rpy changed from rpy="π/2 -π/2 0" to rpy="π/2 0 0"
- leg1joint4 lower and upper limit changed from [-9999, 9999] to [-0.2, 0.2]
- leg1joint[2,6] lower and upper limit changed from [-9999, 9999] to [-π/4, π/4]
- Added 5.95 kg to wheel body and scaled up inertias accordingly[^1]

#### URDF changes for Dragon

- leg4joint[3,5] changed from “continuous” to “fixed”
- leg4joint[2,4,6] changed from “continuous” to “revolute”
- leg4grip[1,2], leg1grip[1,2]bis changed from “prismatic” to “fixed”
- leg4joint4 lower and upper limit changed from [-9999, 9999] to [-0.2, 0.2]
- leg4joint[2,6] lower and upper limit changed from [-9999, 9999] to [-π/4, π/4]
- leg3joint[1-7] changed to from “continuous” to “fixed”
- leg3joint4 origin rpy changed from rpy="π/2 -π/2 0" to rpy="π/2 -6π/7 0"
- leg3joint6 origin rpy changed from rpy="π/2 0 0" to "π/2 -3π/8 0"
- Added 5.95 kg to wheel body and scaled up inertias accordingly


[^1]: I made a quick and dirty Google Sheets to upscale the inertias based on the fact that inertia is directly proportional to mass. You can access this sheet in the drive in the `misc` folder if you for some reason need to scale them (again).