/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#ifndef GB_EXCEPTIONS_HPP
#define GB_EXCEPTIONS_HPP

#include <cstdint>
#include <exception>
#include <vector>
#include <string>

namespace graphblas
{
    //************************************************************************
    class DimensionException : public std::exception
    {
        public:
            DimensionException(std::string const &msg):
                m_message(msg) {}
            DimensionException(){}


        private:
            const char* what() const throw()
            {
                return ("DimensionException: " + m_message).c_str();
            }

            std::string m_message;
    };
} // graphblas

namespace GraphBLAS
{
    //************************************************************************
    class DimensionException : public std::exception
    {
    public:
        DimensionException(std::string const &msg)
            : m_message(msg) {}

        DimensionException(){}

    private:
        const char* what() const throw()
        {
            return ("DimensionException: " + m_message).c_str();
        }

        std::string m_message;
    };

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
