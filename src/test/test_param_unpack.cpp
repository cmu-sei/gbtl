
#include <iostream>

//****************************************************************************
// tags.hpp
//****************************************************************************
namespace grb
{
    // The default matrix is sparse and directed, and the default vector is sparse,
    // so we need tags that modify that
    struct DirectedMatrixTag {};
    struct UndirectedMatrixTag {};
    struct DenseTag {};
    struct SparseTag {};

    namespace detail
    {
        // add category tags in the detail namespace
        struct SparsenessCategoryTag {};
        struct DirectednessCategoryTag {};
        struct NullTag {};
    } //end detail
}//end grb

//****************************************************************************
// backend matrix and vector classes
//****************************************************************************
namespace grb
{
    namespace backend
    {

        template<typename ScalarT>
        class LilSparseMatrix
        {
        };


        template<typename ScalarT>
        class BitmapSparseVector
        {
        };

    } // namespace backend
} // namespace grb

//****************************************************************************
// param_unpack.hpp
//****************************************************************************
namespace grb
{
    namespace backend
    {

        // Substitute template to decide if a tag goes into a given slot
        template<typename TagCategory, typename Tag>
        struct substitute {
            using type = TagCategory;
        };


        template<>
        struct substitute<grb::detail::SparsenessCategoryTag, grb::DenseTag> {
            using type = grb::DenseTag;
        };

        template<>
        struct substitute<grb::detail::SparsenessCategoryTag, grb::SparseTag> {
            using type = grb::SparseTag;
        };

        template<>
        struct substitute<grb::detail::SparsenessCategoryTag, grb::detail::NullTag> {
            using type = grb::SparseTag; // default sparseness
        };


        template<>
        struct substitute<grb::detail::DirectednessCategoryTag, grb::UndirectedMatrixTag> {
            using type = grb::UndirectedMatrixTag;
        };

        template<>
        struct substitute<grb::detail::DirectednessCategoryTag, grb::DirectedMatrixTag> {
            using type = grb::DirectedMatrixTag;
        };

        template<>
        struct substitute<grb::detail::DirectednessCategoryTag, grb::detail::NullTag> {
            //default values
            using type = grb::DirectedMatrixTag; // default directedness
        };


        // hidden part in the frontend (detail namespace somewhere) to unroll
        // template parameter pack

        struct matrix_generator {
            // recursive call: shaves off one of the tags and puts it in the right
            // place (no error checking yet)
            template<typename ScalarT, typename Sparseness, typename Directedness,
                typename InputTag, typename... TagsT>
            struct result {
                using type = typename result<
                    ScalarT,
                    typename substitute<Sparseness, InputTag >::type,
                    typename substitute<Directedness, InputTag >::type,
                    TagsT...>::type;
            };

            //null tag shortcut:
            template<typename ScalarT, typename Sparseness, typename Directedness>
            struct result<ScalarT, Sparseness, Directedness, grb::detail::NullTag, grb::detail::NullTag>
            {
                using type = LilSparseMatrix<ScalarT>;
            };

            // base case returns the matrix from the backend
            template<typename ScalarT, typename Sparseness, typename Directedness, typename InputTag>
            struct result<ScalarT, Sparseness, Directedness, InputTag>
            {
                using type = LilSparseMatrix<ScalarT>;
            };
        };

        // helper to replace backend Matrix class
        template<typename ScalarT, typename... TagsT>
        using Matrix = typename matrix_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            detail::DirectednessCategoryTag,
            TagsT...,
            detail::NullTag,
            detail::NullTag>::type;

        //********************************************************************
        struct vector_generator {
            // recursive call: shaves off one of the tags and puts it in the right
            // place (no error checking yet)
            template<typename ScalarT, typename Sparseness,
                typename InputTag, typename... Tags>
            struct result {
                using type = typename result<
                    ScalarT,
                    typename substitute<Sparseness, InputTag>::type,
                    Tags... >::type;
            };

            // null tag shortcut:
            template<typename ScalarT, typename Sparseness>
            struct result<ScalarT, Sparseness, grb::detail::NullTag>
            {
                using type = BitmapSparseVector<ScalarT>;
            };

            // base case returns the vector from the backend
            template<typename ScalarT, typename Sparseness, typename InputTag>
            struct result<ScalarT, Sparseness, InputTag>
            {
                using type = BitmapSparseVector<ScalarT>;
            };
        };

        // helper to replace backend Vector class
        template<typename ScalarT, typename... TagsT>
        using Vector = typename vector_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            TagsT... ,
            detail::NullTag>::type;

    } // namespace backend
} // namespace grb

//****************************************************************************
// backend eWiseMult operations
//****************************************************************************
namespace grb
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation of 4.3.4.1 eWiseMult: Vector variant
        //**********************************************************************
        template<typename WScalarT,
                 typename... WTagsT>
        inline void eWiseMult(
            grb::backend::BitmapSparseVector<WScalarT>      &w)
            //typename grb::backend::Vector<WScalarT, WTagsT...>       &w)
        {
            std::cout << "eWiseMult<Vector>\n";
        }

        //**********************************************************************
        /// Implementation of 4.3.4.2 eWiseMult: Matrix variant A .* B
        //**********************************************************************
        template<typename CScalarT,
                 typename... CTagsT>
        inline void eWiseMult(
            grb::backend::LilSparseMatrix<CScalarT>         &C)
            //grb::backend::Matrix<CScalarT, CTagsT...>       &C)
        {
            std::cout << "eWiseMult<Matrix>\n";
        }

    } // namespace backend
} // namespace grb

//****************************************************************************
// frontend matrix and vector classes, and eWiseMult operation
//****************************************************************************
namespace grb
{
    //**************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Vector
    {
    public:
        using ScalarType = ScalarT;
        using BackendType = typename backend::Vector<ScalarT, TagsT...>;

    private:
        BackendType m_vec;

        // FRIEND FUNCTIONS

        friend inline BackendType &get_internal_vector(Vector &vector)
        {
            return vector.m_vec;
        }

        friend inline BackendType const &get_internal_vector(Vector const &vector)
        {
            return vector.m_vec;
        }
    };

    //**************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Matrix
    {
    public:
        using ScalarType = ScalarT;
        using BackendType = typename backend::Matrix<ScalarT, TagsT...>;

    private:
        BackendType m_mat;

        // FRIEND FUNCTIONS

        friend inline BackendType &get_internal_matrix(Matrix &matrix)
        {
            return matrix.m_mat;
        }

        friend inline BackendType const &get_internal_matrix(Matrix const &matrix)
        {
            return matrix.m_mat;
        }
    };

    //**************************************************************************
    template<typename WScalarT,
             typename ...WTagsT>
    inline void eWiseMult(Vector<WScalarT, WTagsT...> &w)
    {
        backend::eWiseMult(get_internal_vector(w));
    }
}

//****************************************************************************
int main(int argc, char* argv[])
{
    grb::Vector<bool> equal_flags;

    grb::eWiseMult(equal_flags);

    return 0;
}
