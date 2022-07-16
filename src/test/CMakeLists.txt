
### Make basic tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false *.cpp )

foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message(STATUS "Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} graphblas)
    if (PLATFORM STREQUAL "nwgraph")
      target_link_libraries(${testname} nwgraph)
      target_link_libraries(${testname} TBB::tbb)
    endif()
endforeach( testsourcefile ${TEST_SOURCES} )

### Make extra PLATFORM-specific tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false *.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message(STATUS "Adding: ${testname}_${PLATFORM} ")
    add_executable( ${testname}_${PLATFORM} ${testsourcefile} )#${GRAPHBLAS_HEADERS})
    target_link_libraries(${testname}_${PLATFORM} graphblas)
    if (PLATFORM STREQUAL "nwgraph")
      target_link_libraries(${testname}_${PLATFORM} nwgraph)
      target_link_libraries(${testname}_${PLATFORM} TBB::tbb)
    endif()
endforeach( testsourcefile ${TEST_SOURCES} )
