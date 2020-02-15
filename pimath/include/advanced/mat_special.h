#pragma once
/*
    This file defines special operations that might be useful under certain situations.
    Like transform a Vector3 by a Matrix4x4
*/

#include "../pimath.h"

PIMATH_NAMESPACE_BEGIN
namespace MatrixSpecial {

    /*
        applyTransform, used for multiplication between Matrix4x4 and Vector3
        the parameter: w_comp can be used to control if a point or a direction is being transformed.
    */

    template<typename T, InstSetExt ISE,
        typename std::enable_if_t<std::is_same<T, float32>::value && ISE >= InstSetExt::SSE, int> = 0>
        PM_INLINE MatrixND<3, 1, T, ISE> applyTransform(const MatrixND<4, 4, T, ISE>& mat, const MatrixND<3, 1, T, ISE>& vec,
            T w_comp = T(1))
    {
        __m128 v = _mm_setr_ps(vec.x, vec.y, vec.z, w_comp);
        MatrixND<4, 1, T, ISE> compVec; compVec.data[0].v = v;
        MatrixND<4, 1, T, ISE> result = mat * compVec;
        if (result.w == T(0))
            return MatrixND<3, 1, T, ISE>(result.x, result.y, result.z);
        return MatrixND<3, 1, T, ISE>(result.x / result.w, result.y / result.w, result.z / result.w);
    }

    template<typename T, InstSetExt ISE,
        typename std::enable_if_t<!std::is_same<T, float32>::value || ISE < InstSetExt::SSE, int> = 0>
        PM_INLINE MatrixND<3, 1, T, ISE> applyTransform(const MatrixND<4, 4, T, ISE> & mat, const MatrixND<3, 1, T, ISE> & vec,
            T w_comp = T(1))
    {
        T result[4];
        for (int r = 0; r < 4; ++r) {
            T v = T(0);
            for (int c = 0; c < 3; ++c) {
                v = v + mat(r, c) * vec(c, 0);
            }
            v = v + mat(r, 3) * w_comp;
            result[r] = v;
        }
        if (result[3] == T(0))
            return MatrixND<3, 1, T, ISE>(result[0], result[1], result[2]);
        return MatrixND<3, 1, T, ISE>(result[0] / result[3], result[1] / result[3], result[2] / result[3]);
    }

    /*
        Multiply two matrix Colume by Colume.
        It's useful for normal transformation
    */
    template<typename T, InstSetExt ISE,
            typename std::enable_if_t<std::is_same<T, float32>::value && ISE >= InstSetExt::SSE, int> = 0 >
    PM_INLINE MatrixND<3, 1, T, ISE> NormalTransform(const MatrixND<4, 4, T, ISE>& mat, const MatrixND<3, 1, T, ISE>& vec)
    {
        T result[4];
        __m128 v = _mm_setr_ps(vec.x, vec.y, vec.z, T(0));
        for (int r = 0; r < 4; ++r) {
            _mm_store_ss(&result[r], _mm_dp_ps(mat.data[r].v, v, 0xf1));
        }
        if (result[3] == T(0))
            return MatrixND<3, 1, T, ISE>(result[0], result[1], result[2]);
        return MatrixND<3, 1, T, ISE>(result[0] / result[3], result[1] / result[3], result[2] / result[3]);
    }

    template<typename T, InstSetExt ISE,
        typename std::enable_if_t<!std::is_same<T, float32>::value || ISE < InstSetExt::SSE, int> = 0 >
        PM_INLINE MatrixND<3, 1, T, ISE> NormalTransform(const MatrixND<4, 4, T, ISE>& mat, const MatrixND<3, 1, T, ISE>& vec)
    {
        T result[3];
        for (int c = 0; c < 3; ++c) {
            result[c] = T(0);
            for (int i = 0; i < 3; ++i) {
                result[c] = mat.data[c][i] * vec.data[0][i];
            }
        }
         return MatrixND<3, 1, T, ISE>(result[0], result[1], result[2]);
    }
}
PIMATH_NAMESPACE_END