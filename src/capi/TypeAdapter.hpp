//
// Created by aomellinger on 5/17/17.
//

#ifndef SRC_TYPEADAPTER_H
#define SRC_TYPEADAPTER_H

#include <iostream>

#include "TypeUnion.hpp"

/**
 * This class is basically a dynamic type that selects automtically
 * during runtimne between the different types. It is used as a bridge
 * to adapt the back-end C++ implementation.
 */

class TypeAdapter
{
public:

    TypeAdapter()
    : m_type(GrB_Unset_Type)
    {
        m_val._int32 = 0;
        //std::cout << "Constructed via empty (any)" << std::endl;
    }

    // We can construct from all basic types
    TypeAdapter(GrB_BOOL val)
    : m_type(GrB_BOOL_Type)
    {
        m_val._bool = val;
        //std::cout << "Constructed via BOOl" << std::endl;
    }

    TypeAdapter(GrB_INT32 val)
    : m_type(GrB_INT32_Type)
    {
        m_val._int32 = val;
        //std::cout << "Constructed via int32" << std::endl;
    }

    TypeAdapter(GrB_FP32 val)
    : m_type(GrB_FP32_Type)
    {
        m_val._fp32 = val;
        //std::cout << "Constructed via fp32" << std::endl;
    }

    TypeAdapter(GrB_FP64 val)
    : m_type(GrB_FP64_Type)
    {
        m_val._fp64 = val;
        //std::cout << "Constructed via fp64" << std::endl;
    }

    TypeAdapter(GrB_Type type)
            : m_type(type)
    {
        //std::cout << "Typed constructor: " << m_type << std::endl;
    }

    TypeAdapter(GrB_Type type, const GrB_TypeUnion &value)
    : m_type(type), m_val(value)
    {
        //std::cout << "Type/value constructor: " << m_type << std::endl;
    }

    TypeAdapter(const TypeAdapter& rhs)
    : m_type(rhs.m_type), m_val(rhs.m_val)
    {
        //std::cout << "Copy constructor: " << m_type << std::endl;
    }

//    TypeAdapter(GrB_Type type, const void *value)
//            : m_type(type)
//    {
//        std::cout << "++++++++++++++++++++++++++ WTF +++++++++++++++ " << std::endl;
//        switch (type)
//        {
//            case GrB_BOOL_Type:break;
//            case GrB_INT32_Type:break;
//            case GrB_FP32_Type:break;
//            case GrB_FP64_Type:break;
//        }
//    }

    TypeAdapter &operator=(const TypeAdapter &rhs)
    {
        // This should invoke the other casting operator (below) and pull
        // the value out as the correct thing.  We then set out value based
        // on that cast.
        if (m_type == GrB_Unset_Type)
        {
            //std::cout << "assuming other type" << std::endl;
            if (rhs.m_type == GrB_Unset_Type)
                std::cerr << "!!!! Why are we set equal to an unset type???" << std::endl;
            m_type = rhs.m_type;
        }

        switch (m_type)
        {
            case GrB_Unset_Type:
                // This can't happen because it would have been set from above.
                m_val._int32 = 0;
                break;
            case GrB_BOOL_Type:
                m_val._bool     = static_cast<GrB_BOOL>(rhs);
                break;
            case GrB_INT32_Type:
                m_val._int32    = static_cast<GrB_INT32>(rhs);
                break;
            case GrB_FP32_Type:
                m_val._fp32     = static_cast<GrB_FP32>(rhs);
                break;
            case GrB_FP64_Type:
                m_val._fp64     = static_cast<GrB_FP64>(rhs);
                break;
        }

        return *this;
    }

    // Da magic is here!
    template < typename T>
    operator T() const
    {
        switch (m_type)
        {
            case GrB_Unset_Type:    return static_cast<T>(0);
            case GrB_BOOL_Type:     return static_cast<T>(m_val._bool);
            case GrB_INT32_Type:    return static_cast<T>(m_val._int32);
            case GrB_FP32_Type:     return static_cast<T>(m_val._fp32);
            case GrB_FP64_Type:     return static_cast<T>(m_val._fp64);
        }

        // @TODO: We should never get here.  Find some useful error message.
        printf("!!!! TypeAdapter::operator T() encountered unkown type!!\n");
        return (T) 0;
    }

    // @TOOO: Get a better name for this
    void *getUnion()
    {
        return &m_val;
    }

    /** Converts the TypeAdapter to a new type. */
    TypeAdapter convert(GrB_Type newType)
    {
        switch (newType)
        {
            case GrB_Unset_Type:
                std::cerr << "Why are we casting to UNSET??" << std::endl;
                return TypeAdapter();
            case GrB_BOOL_Type:  return TypeAdapter(static_cast<GrB_BOOL >(*this));
            case GrB_INT32_Type: return TypeAdapter(static_cast<GrB_INT32 >(*this));
            case GrB_FP32_Type:  return TypeAdapter(static_cast<GrB_FP32>(*this));
            case GrB_FP64_Type:  return TypeAdapter(static_cast<GrB_FP64>(*this));
        }

        // @TODO: We should never get here.  Find some useful error message.
        printf("!!!! TypeAdapter::convert encountered unkown type!!\n");
        return TypeAdapter((GrB_FP64) 0);
    }

    // Is there a way we can force the construction of the above?
    //template <> operator GrB_FP64<GrB_FP64>();

    friend std::ostream& operator<< (std::ostream &os, TypeAdapter const &adapter)
    {
        switch (adapter.m_type)
        {
            case GrB_Unset_Type:    os << "Unset";  break;
            case GrB_BOOL_Type:     os << std::to_string(adapter.m_val._bool);  break;
            case GrB_INT32_Type:    os << std::to_string(adapter.m_val._int32) << "i"; break;
            case GrB_FP32_Type:     os << std::to_string(adapter.m_val._fp32) << "f";  break;
            case GrB_FP64_Type:     os << std::to_string(adapter.m_val._fp64) << "d";  break;
        }
        return os;
    }

private:

    // The type we are
    GrB_Type m_type;

    // The pointer to the value
    GrB_TypeUnion   m_val;
};

#endif //SRC_TYPEADAPTER_H
