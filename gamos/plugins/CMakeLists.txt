cmake_minimum_required(VERSION 3.16)
project(gamma_camera_mc_plugins CXX)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

# GAMOS is located through the GAMOS_INSTALL environment variable; see
# docs/install.md. The plug-in is registered with the GAMOS user-action factory
# at load time.
include_directories(include $ENV{GAMOS_INSTALL}/include)

add_library(PhotonHistoryUA SHARED src/PhotonHistoryUA.cc)
target_link_libraries(PhotonHistoryUA ${Geant4_LIBRARIES})
target_compile_features(PhotonHistoryUA PUBLIC cxx_std_17)
