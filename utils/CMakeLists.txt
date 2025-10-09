project(cpu_utils)

find_package(Boost REQUIRED COMPONENTS fiber context)

include_directories(${Boost_INCLUDE_DIRS})
add_library(cpu_utils SHARED cpu_utils.cpp)
target_link_libraries(cpu_utils PRIVATE ${Boost_LIBRARIES})
