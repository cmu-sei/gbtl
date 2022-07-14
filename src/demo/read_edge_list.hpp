/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2021 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <iostream>
#include <fstream>
#include <graphblas/graphblas.hpp>

using namespace grb;

//****************************************************************************
IndexType read_edge_list(std::string const &pathname,
                         IndexArrayType    &row_indices,
                         IndexArrayType    &col_indices)
{
    std::ifstream infile(pathname);
    IndexType max_id = 0;
    IndexType min_id = 1; // Assuming 1 or 0 based indexing; nothing else
    uint64_t num_rows = 0;
    uint64_t src, dst;

    std::string line;
    // First pass to get min ID
    while (std::getline( infile, line) )
    {
        if (infile.eof()) break;

        std::istringstream l(line);
        l >> src >> dst; // And discard the rest (weights)
        min_id = std::min(min_id, std::min(src, dst));
    }
    infile.clear(); // Reset EOF flag
    infile.seekg(0, std::ios_base::beg); // Reset infile to start
    // std::cout << "Min vertex ID: " << min_id << std::endl;

    while (std::getline( infile, line) )
    {
        if (infile.eof()) break;

        std::istringstream l(line);
        l >> src >> dst; // And discard the rest (weights)
        src -= min_id;
        dst -= min_id;

        //std::cout << "Read: " << src << ", " << dst << std::endl;
        max_id = std::max(max_id, src);
        max_id = std::max(max_id, dst);

        //if (src > max_id) max_id = src;
        //if (dst > max_id) max_id = dst;

        row_indices.push_back(src);
        col_indices.push_back(dst);

        ++num_rows;
    }
    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    return (max_id + 1);
}
