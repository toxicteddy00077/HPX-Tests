cmake_minimum_required(VERSION 3.19)
project(HPX_MatMul CXX)

set(HPX_DIR "/opt/hpx/lib/cmake/HPX")
set(Tbb_LIBRARY "/usr/lib/x86_64-linux-gnu/libtbbmalloc.so" )

find_package(HPX REQUIRED)
find_package(TBB REQUIRED MODULE)

add_executable(MatMul MatMul.cpp)
target_link_libraries(MatMul PRIVATE 
    HPX::hpx 
    HPX::wrap_main 
    HPX::iostreams_component 
    ${TBB_LIBRARIES} 
    ${TBBmalloc_LIBRARY} 
    -ltbb
)
