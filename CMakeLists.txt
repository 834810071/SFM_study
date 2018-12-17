cmake_minimum_required(VERSION 3.13)
project(SFM_study)

set(CMAKE_CXX_STANDARD 14)

add_executable(SFM_study main.cpp)

set(Eigen_INCLUDE_DIRS "/usr/local/include/eigen3")
include_directories(${Eigen_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)

INCLUDE_DIRECTORIES(${GTSAM_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

target_link_libraries(SFM_study ${OpenCV_LIBS} ${Eigen_INCLUDE_DIRS} -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_imgcodecs -lgtsam -lboost_system -ltbb)