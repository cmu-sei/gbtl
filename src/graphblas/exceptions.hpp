/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#ifndef GB_EXCEPTIONS_HPP
#define GB_EXCEPTIONS_HPP

#include <cstdint>
#include <exception>
#include <vector>
#include <string>
#include <sstream>
#include <graphblas/detail/logging.h>

namespace GraphBLAS
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

#endif // GB_EXCEPTIONS_HPP
