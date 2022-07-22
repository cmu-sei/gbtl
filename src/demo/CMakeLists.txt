

## Make demos
file( GLOB TEST_SOURCES LIST_DIRECTORIES false *.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message(STATUS "Adding: ${testname} ")
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} graphblas)

    if (PLATFORM STREQUAL "nwgraph")
      target_link_libraries(${testname} nwgraph)
      target_link_libraries(${testname} TBB::tbb)
    endif()
endforeach( testsourcefile ${TEST_SOURCES} )

