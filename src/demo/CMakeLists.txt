
## Make demos
# file( GLOB TEST_SOURCES LIST_DIRECTORIES false *.cpp )
# foreach( testsourcefile ${TEST_SOURCES} )
#     get_filename_component(justname ${testsourcefile} NAME)
#     string( REPLACE ".cpp" "" testname ${justname} )
#     message("Adding: ${testname}")
#     add_executable( ${testname} ${testsourcefile} ${GRAPHBLAS_HEADERS})
#     target_link_libraries( ${testname} ${TBB_LIBRARIES} )
#     target_include_directories(
#       ${testname}
#       PUBLIC ${CMAKE_SOURCE_DIR}/src
#       PRIVATE ${PLATFORM_SOURCE_DIR}
#       )
#     if (PLATFORM STREQUAL "nwgraph")
#       target_include_directories(
#         ${testname}
#         PUBLIC ${TBB_INCLUDE_DIRS}/src
#         PUBLIC ${NWGRAPH_INCLUDE_DIRECTORIES}
#         )
#     endif()
# endforeach( testsourcefile ${TEST_SOURCES} )
