
### Make basic tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false *.cpp )

foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message(STATUS "Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} )# ${GRAPHBLAS_HEADERS})

#  message(STATUS "+++ GRAPHBLAS_HEADERS=${GRAPHBLAS_HEADERS}")

    target_include_directories(
      ${testname}
      PUBLIC ${CMAKE_SOURCE_DIR}/src
      PRIVATE ${PLATFORM_SOURCE_DIR}
      )

    target_link_libraries( ${testname} graphblas ${TBB_LIBRARIES} )

get_target_property(dirs ${testname} INTERFACE_INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "+++ dir='${dir}'")
endforeach()

    if (PLATFORM STREQUAL "nwgraph")
      target_include_directories(
        ${testname}
        PUBLIC ${TBB_INCLUDE_DIRS}
        PUBLIC ${NWGRAPH_INCLUDE_DIRECTORIES}
        )
    endif()
endforeach( testsourcefile ${TEST_SOURCES} )

### Make extra PLATFORM-specific tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false *.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}_${PLATFORM} ")
    add_executable( ${testname}_${PLATFORM} ${testsourcefile} ${GRAPHBLAS_HEADERS})
    target_link_libraries( ${testname}_${PLATFORM} ${TBB_LIBRARIES} )
    target_include_directories(
      ${testname}_${PLATFORM}
      PUBLIC ${CMAKE_SOURCE_DIR}/src
      private ${PLATFORM_SOURCE_DIR}
      )
    if (PLATFORM STREQUAL "nwgraph")
      target_include_directories(
        ${testname}_${PLATFORM}
        PUBLIC ${TBB_INCLUDE_DIRS}
        PUBLIC ${NWGRAPH_INCLUDE_DIRECTORIES}
        )
    endif()
endforeach( testsourcefile ${TEST_SOURCES} )
