project(cpu_utils)

find_package(Boost REQUIRED COMPONENTS fiber context) 

include_directories(${Boost_INCLUDE_DIRS}) 
add_library(cpu_utils SHARED cpu_utils.cpp)
target_link_libraries(cpu_utils PRIVATE ${Boost_LIBRARIES})

# if(APPLE)
#  target_link_options(cpu_utils PRIVATE "LINKER:--undefined,dynamic_lookup")
# elseif(UNIX)
#  target_link_options(cpu_utils PRIVATE "LINKER:--exclude-libs,ALL")
# elseif(WIN32)
#  target_link_options(cpu_utils PRIVATE "LINKER:--enable-auto-import,--enable-runtime-pseudo-reloc")
# endif()

