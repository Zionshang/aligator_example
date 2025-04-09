cmake_minimum_required(VERSION 3.5.0)
project(aligator_example VERSION 0.1.0 LANGUAGES C CXX)

find_package(aligator REQUIRED)
add_executable(se2_car se2_car.cpp)
add_executable(talos_walk talos_walk.cpp)
add_executable(solo_kinodynamics solo_kinodynamics.cpp)
add_executable(mpc mpc.cpp)

target_link_libraries(se2_car aligator::aligator)
target_link_libraries(talos_walk aligator::aligator)
target_link_libraries(solo_kinodynamics aligator::aligator)
target_link_libraries(mpc aligator::aligator)

configure_file (
  "${PROJECT_SOURCE_DIR}/utils/project_path.hpp.in"
  "${PROJECT_SOURCE_DIR}/utils/project_path.hpp" @ONLY
)