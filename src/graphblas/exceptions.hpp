/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
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

#include <cstdint>
#include <exception>
#include <vector>
#include <string>
#include <sstream>
#include <graphblas/detail/logging.h>

namespace grb
{
    //************************************************************************
    // Execution Errors, Table 2.5(b) of Spec version 1.0.2
    //************************************************************************

    // out of memory handled by language

    //************************************************************************
    class IndexOutOfBoundsException : public std::exception
    {
    public:
        IndexOutOfBoundsException(std::string const &msg)
            : m_message(msg) {}

        IndexOutOfBoundsException() {}

    private:
        const char* what() const throw()
        {
            return ("IndexOutOfBoundsException: " + m_message).c_str();
        }

        std::string m_message;
    };

    //************************************************************************
    class PanicException : public std::exception
    {
    public:
        PanicException(std::string const &msg)
            : m_message(msg) {}

        PanicException() {}

    private:
        const char* what() const throw()
        {
            return ("PanicException: " + m_message).c_str();
        }

        std::string m_message;
    };

    //************************************************************************
    // API Errors, Table 2.5(a) of Spec version 1.0.2
    //************************************************************************

    //************************************************************************
    class InvalidValueException : public std::exception
    {
    public:
        InvalidValueException(std::string const &msg)
            : m_message(msg) {}

        InvalidValueException() {}

    private:
        const char* what() const throw()
        {
            return ("InvalidValueException: " + m_message).c_str();
        }

        std::string m_message;
    };

    //************************************************************************
    class InvalidIndexException : public std::exception
    {
    public:
        InvalidIndexException(std::string const &msg)
            : m_message(msg) {}

        InvalidIndexException() {}

    private:
        const char* what() const throw()
        {
            return ("InvalidIndexException: " + m_message).c_str();
        }

        std::string m_message;
    };

    //************************************************************************
    class DimensionException : public std::exception
    {
    public:
        DimensionException(std::string const &msg)
            : m_message(msg)
        {
            GRB_LOG_VERBOSE("!!! DimensionException: " << msg);
        }

        DimensionException(){}

    private:
        const char* what() const throw()
        {
            return ("DimensionException: " + m_message).c_str();
        }

        std::string m_message;
    };

    //************************************************************************
    class OutputNotEmptyException : public std::exception
    {
    public:
        OutputNotEmptyException(std::string const &msg)
            : m_message(msg)
        {
            GRB_LOG_VERBOSE("!!! OutputNotEmptyException: " << msg);
        }

        OutputNotEmptyException(){}

    private:
        const char* what() const throw()
        {
            return ("OutputNotEmptyException: " + m_message).c_str();
        }

        std::string m_message;
    };

    //************************************************************************
    class NoValueException : public std::exception
    {
    public:
        NoValueException(std::string const &msg)
            : m_message(msg) {}

        NoValueException() {}

    private:
        const char* what() const throw()
        {
            return ("NoValueException: " + m_message).c_str();
        }

        std::string m_message;
    };
}
