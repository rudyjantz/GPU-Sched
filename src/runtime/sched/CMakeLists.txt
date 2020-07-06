add_executable(bemps_sched 
    sched.cpp)

find_library(LIBRT rt HINTS /usr/local/cuda/lib64)

include_directories(
    ${SOURCE_DIR}/runtime/bemps
)

target_link_libraries(bemps_sched ${LIBRT} bemps pthread)

if (GPU_RES)
    add_definitions(-D${GPU_RES})
endif()
