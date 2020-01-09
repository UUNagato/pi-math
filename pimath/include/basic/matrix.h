#pragma once
#include <memory>
#include <functional>
#include <algorithm>
#include <immintrin.h>
#include <type_traits>
#include <initializer_list>

#include "utils.h"
#include "datatype.h"
#include "array.h"

NAMESPACE_PIMATH_BEGIN

template<int rows, int cols, typename T, InstSetExt ISE>
struct MatrixBase
{
    ArrayND<rows, T, ISE> data[cols];
};

// special accessor for Vector-like Matrix
template<typename T, InstSetExt ISE>
struct MatrixBase<1, 1, T, ISE>
{
    MatrixBase() {}
    union {
        ArrayND<1, T, ISE> data[1];
        T x;
        T r;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<2, 1, T, ISE>
{
    MatrixBase() {}
    union {
		ArrayND<2, T, ISE> data[1];
        T x, y;
        T r, g;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<3, 1, T, ISE>
{
    MatrixBase() {}
    union {
		ArrayND<3, T, ISE> data[1];
        T x, y, z;
        T r, g, b;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<4, 1, T, ISE>
{
    MatrixBase() {}
    union {
		ArrayND<4, T, ISE> data[1];
        T x, y, z, w;
        T r, g, b, a;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 2, T, ISE>
{
    MatrixBase() {}
    union {
		ArrayND<1, T, ISE> data[2];
        T x, y;
        T r, g;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 3, T, ISE>
{
    MatrixBase() {}
    union {
		ArrayND<1, T, ISE> data[3];
        T x, y, z;
        T r, g, b;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 4, T, ISE>
{
    MatrixBase() {}
    union {
		ArrayND<1, T, ISE> data[4];
        T x, y, z, w;
        T r, g, b, a;
    };
};

template<int rows, int cols, typename T, InstSetExt ISE = default_instruction_set>
struct MatrixND : public MatrixBase<rows, cols, T, ISE>
{
    static constexpr bool IS_VECTOR = (cols == 1) || (rows == 1);

    template<int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    static constexpr bool MATRIX_SSE = (cols_ <= 4) && std::is_same<T_, float32>::value && ISE_ >= InstSetExt::SSE;

    // Constructors
    MatrixND() {
    }

    // Single value Constructor
    MatrixND(T v) {
        for (int i = 0; i < cols; ++i)
            for (int j = 0; j < rows; ++j) {
                this->data[i][j] = v;
            }
    }

    // Copy Constructor
    MatrixND(const MatrixND& v) {
        memcpy(this, &v, sizeof(*this));
    }

    // Function Constructor
    template<typename F, std::enable_if_t<std::is_convertible<F, std::function<T(int, int)>>::value, int> = 0>
    explicit MatrixND(const F& f) {
        for (int i = 0; i < cols; ++i)
            for (int j = 0; j < rows; ++j) {
                this->data[i][j] = f(j, i);
            }
    }

    template<typename F, std::enable_if_t<std::is_convertible<F, std::function<ArrayND<rows, T, ISE>(int)>>::value, int> = 0>
    explicit MatrixND(const F& f) {
        for (int i = 0; i < cols; ++i)
            this->data[i] = f(i);
    }

    // Variant Value Constructor
    explicit MatrixND(const std::initializer_list<T>& list) : MatrixND() {
        int r = 0, c = 0;
        for (auto iter : list) {
            this->data[c][r] = iter;
            if (++c >= cols) {
                c = 0;
                if (++r >= rows)
                    break;
            }
        }
    }

    // Special Vector Constructor
    explicit MatrixND(T x, T y) {
        static_assert((rows == 1 && cols == 2) || (rows == 2 && cols == 1), "Matrix must be a 2-dimensional vector");
        this->x = x;
        this->y = y;
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!ArrayND<rows_, T_, ISE_>::SIMD_FLAG, int> = 0>
        explicit MatrixND(T x, T y, T z) {
        static_assert((rows == 1 && cols == 3) || (rows == 3 && cols == 1), "Matrix must be a 3-dimensional vector");
        this->x = x;
        this->y = y;
        this->z = z;
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<ArrayND<rows_, T_, ISE_>::SIMD_FLAG, int> = 0>
        explicit MatrixND(T x, T y, T z) {
        static_assert(rows == 3 && cols == 1, "Matrix must be a 3-dimensional vector");
        this->data[0].v = _mm_set_ps(x, y, z, 0f);
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!ArrayND<rows_, T_, ISE_>::SIMD_FLAG, int> = 0>
        explicit MatrixND(T x, T y, T z, T w) {
        static_assert((rows == 1 && cols == 4) || (rows == 4 && cols == 1), "Matrix must be a 4-dimensional vector");
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<ArrayND<rows_, T_, ISE_>::SIMD_FLAG, int> = 0>
        explicit MatrixND(T x, T y, T z, T w) {
        static_assert(rows == 4 && cols == 1, "Matrix must be a 3-dimensional vector");
        this->data[0].v = _mm_set_ps(x, y, z, w);
    }

    // Special Matrix initializer
    template<int rows_ = rows>
    explicit MatrixND(const std::initializer_list<ArrayND<rows_, T, ISE>>& list) {
        static_assert(rows_ == rows, "Matrix must have exactly same number of rows with the input array length");
        int c = 0;
        for (auto iter : list) {
            if (c >= cols)
                return;
            this->data[c] = iter;
            ++c;
        }
    }

    // some basic operators
    PM_INLINE ArrayND<rows, T, ISE>& operator[] (size_t index) {
        return this->data[index];
    }

    PM_INLINE const ArrayND<rows, T, ISE>& operator[] (size_t index) const {
        return this->data[index];
    }

    PM_INLINE T& operator() (size_t row, size_t col) {
        return this->data[col][row];
    }

    PM_INLINE const T operator() (size_t row, size_t col) const {
        return this->data[col][row];
    }

    PM_INLINE MatrixND& operator= (const MatrixND& m) {
        memcpy(this, &m, sizeof(*this));
        return *this;
    }

    PM_INLINE bool operator== (const MatrixND& m) {
        for (int i = 0; i < cols; ++i)
            if (!(this->data[i] == m[i]))
                return false;
        return true;
    }

    // Arithmatic
    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND operator+ (const MatrixND& m) {
        return MatrixND([=](int c) { return this->data[c] + m[c]; });
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND& operator+= (const MatrixND& m) {
        for (int c = 0; c < cols; ++c)
            this->data[c] = this->data[c] + m[c];
        return *this;
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND operator- (const MatrixND& m) {
        return MatrixND([=](int c) { return this->data[c] - m[c]; });
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND& operator-= (const MatrixND& m) {
        for (int c = 0; c < cols; ++c)
            this->data[c] = this->data[c] - m[c];
        return *this;
    }

    // =======================================================================================
    // Multiplication
    // =======================================================================================
    template<int lrows = rows, int lcols = cols, int rrows = cols, int rcols, typename T_ = T,
        InstSetExt ISE_ = ISE>
        static void multiply(const MatrixND<lrows, lcols, T_, ISE_>& m1,
            const MatrixND<rrows, rcols, T_, ISE_>& m2, MatrixND<lrows, rcols, T_, ISE_>* ret)
    {
        static_assert(lcols == rrows, "The multiplication matrix have incompatible dimensions.");
        for (int r = 0; r < lrows; ++r) {
            for (int c = 0; c < rcols; ++c) {
                T tmp = T();
                for (int i = 0; i < lcols; ++i) {
                    tmp = tmp + m1(r, i) * m2(i, c);
                }
                (*ret)(r, c) = tmp;
            }
        }
    }

    template<int lrows = rows, int lcols = cols, int rrows = cols, int rcols, typename T_ = T,
        InstSetExt ISE_ = ISE>
        static void multiply_SSE(const MatrixND<lrows, lcols, T_, ISE_>& m1,
            const MatrixND<rrows, rcols, T_, ISE_>& m2, MatrixND<lrows, rcols, T_, ISE_>* ret)
    {
        static_assert(lcols == rrows, "The multiplication matrix have incompatible dimensions.");
        // pack first matrix rows into four __m128
        float r[4][4] = { 0 };
        for (int i = 0; i < lrows; ++i)
            for (int j = 0; j < lcols; ++j)
                r[j][i] = m1(i, j);
        __m128 rr[4];
        rr[0] = _mm_load_ps(r[0]);
        rr[1] = _mm_load_ps(r[1]);
        rr[2] = _mm_load_ps(r[2]);
        rr[3] = _mm_load_ps(r[3]);

        for (int r = 0; r < lrows; ++r) {
            for (int c = 0; c < rcols; ++c) {
                _mm_store_ss(&(ret->operator()(r, c)), _mm_dp_ps(rr[r], m2[c].v, 0xf1));
            }
        }
    }

    template<int cols_ = cols, int rrows, int rcols,
        typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!MATRIX_SSE<cols_, T, ISE> || !MATRIX_SSE<rrows, T_, ISE_>, int> = 0>
        PM_INLINE MatrixND<rows, rcols, T, ISE> operator* (const MatrixND<rrows, rcols, T_, ISE_>& m2)
    {
        MatrixND<rows, rcols, T, ISE> ret;
        multiply(*this, m2, &ret);
        return ret;
    }

    template<int cols_ = cols, int rrows, int rcols,
        typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<MATRIX_SSE<cols_, T, ISE> && MATRIX_SSE<rrows, T_, ISE_>, int> = 0>
        PM_INLINE MatrixND<rows, rcols, T, ISE> operator* (const MatrixND<rrows, rcols, T_, ISE_>& m2)
    {
        // std::cout << "SSE used" << std::endl;
        MatrixND<rows, rcols, T, ISE> ret;
        multiply_SSE(*this, m2, &ret);
        return ret;
    }

    // ================================================================================
    // Scalar
    // ================================================================================
    PM_INLINE MatrixND operator+ (const T scalar) {
        return MatrixND([=](int c) { return this->data[c] + scalar; });
    }

    PM_INLINE MatrixND operator- (const T scalar) {
        return MatrixND([=](int c) { return this->data[c] - scalar; });
    }

    PM_INLINE MatrixND operator* (const T scalar) {
        return MatrixND([=](int c) { return this->data[c] * scalar; });
    }

    PM_INLINE MatrixND operator/ (const T scalar) {
        return MatrixND([=](int c) { return this->data[c] / scalar; });
    }

    PM_INLINE MatrixND& operator+= (const T scalar) {
        for (int c = 0; c < cols; ++c)
            this->data[c] += scalar;
        return *this;
    }

    PM_INLINE MatrixND& operator-= (const T scalar) {
        for (int c = 0; c < cols; ++c)
            this->data[c] -= scalar;
        return *this;
    }

    PM_INLINE MatrixND& operator*= (const T scalar) {
        for (int c = 0; c < cols; ++c)
            this->data[c] *= scalar;
        return *this;
    }

    PM_INLINE MatrixND& operator/= (const T scalar) {
        for (int c = 0; c < cols; ++c)
            this->data[c] /= scalar;
        return *this;
    }

    // ===============================================================================
    // Operations for matrices
    // ===============================================================================
    PM_INLINE MatrixND<cols, rows, T, ISE> transpose() {
        return MatrixND<cols, rows, T, ISE>([=](int r, int c) { return (*this)(c, r); });
    }

    PM_INLINE T determinant() {
        static_assert(rows == cols, "Determinant can only be calculated on squared matrix");

    }

	// ===============================================================================
	// Operations for vectors
	// ===============================================================================
	PM_INLINE T length2() {
		T total_sqr = T(0);
		for (int c = 0; c < cols; ++c)
			for (int r = 0; r < rows; ++r)
				total_sqr += (*this)(r, c) * (*this)(r, c);
		return total_sqr;
	}

	PM_INLINE T length() {
		T sqrLen = length2();
		return sqrt(sqrLen);
	}

	PM_INLINE MatrixND normalize() {
        T len = length();
        if (len <= 0.00001f && len >= -0.00001f)
            return MatrixND(T(0));
        else
            return (*this) / len;
	}
};

// IO for matrix
template<int rows, int cols, typename T, InstSetExt ISE>
std::ostream& operator<<(std::ostream &os, const MatrixND<rows, cols, T, ISE>&v)
{
    os << '[';
    for (int r = 0; r < rows - 1; ++r) {
        for (int c = 0; c < cols; ++c) {
            os << v(r, c) << ',';
        }
        os << '\n';
    }

    // the last row
    for (int c = 0; c < cols - 1; ++c)
        os << v(rows - 1, c) << ',';
    os << v(rows - 1, cols - 1) << ']' << std::endl;
    return os;
}

//======================================================================
// Operators for Matrices
//======================================================================

//======================================================================
// Determinant
//======================================================================
template<typename T, InstSetExt ISE>
T determinant(const MatrixND<2, 2, T, ISE>& mat) {
    return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
}

template<typename T, InstSetExt ISE>
T determinant(const MatrixND<3, 3, T, ISE>& mat) {
    return mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) -
        mat[1][0] * (mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2]) +
        mat[2][0] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
}

template <typename T, InstSetExt ISE>
T determinant(const MatrixND<4, 4, T, ISE>& m) {
    // This function is adopted from GLM
    /*
    ================================================================================
    OpenGL Mathematics (GLM)
    --------------------------------------------------------------------------------
    GLM is licensed under The Happy Bunny License and MIT License
    ================================================================================
    The Happy Bunny License (Modified MIT License)
    --------------------------------------------------------------------------------
    Copyright (c) 2005 - 2014 G-Truc Creation
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    Restrictions:
     By making use of the Software for military purposes, you choose to make a
     Bunny unhappy.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    ================================================================================
    The MIT License
    --------------------------------------------------------------------------------
    Copyright (c) 2005 - 2014 G-Truc Creation
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
     */

    T Coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    T Coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
    T Coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

    T Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    T Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
    T Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

    T Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    T Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
    T Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

    T Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    T Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
    T Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

    T Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    T Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
    T Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

    T Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
    T Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
    T Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

    using Vector = ArrayND<4, T, ISE>;

    Vector Fac0(Coef00, Coef00, Coef02, Coef03);
    Vector Fac1(Coef04, Coef04, Coef06, Coef07);
    Vector Fac2(Coef08, Coef08, Coef10, Coef11);
    Vector Fac3(Coef12, Coef12, Coef14, Coef15);
    Vector Fac4(Coef16, Coef16, Coef18, Coef19);
    Vector Fac5(Coef20, Coef20, Coef22, Coef23);

    Vector Vec0(m[1][0], m[0][0], m[0][0], m[0][0]);
    Vector Vec1(m[1][1], m[0][1], m[0][1], m[0][1]);
    Vector Vec2(m[1][2], m[0][2], m[0][2], m[0][2]);
    Vector Vec3(m[1][3], m[0][3], m[0][3], m[0][3]);

    Vector Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
    Vector Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
    Vector Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
    Vector Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

    Vector SignA(+1, -1, +1, -1);
    Vector SignB(-1, +1, -1, +1);
    MatrixND<4, 4, T, ISE> Inverse{ Inv0 * SignA, Inv1 * SignB, Inv2 * SignA,
        Inv3 * SignB };

    Vector Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

    Vector Dot0(m[0] * Row0);
    T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

    return Dot1;
}

template <typename T, InstSetExt ISE>
MatrixND<4, 4, T, ISE> inversed(const MatrixND<4, 4, T, ISE>& m) {
    // This function is copied from GLM
    /*
    ================================================================================
    OpenGL Mathematics (GLM)
    --------------------------------------------------------------------------------
    GLM is licensed under The Happy Bunny License and MIT License
    ================================================================================
    The Happy Bunny License (Modified MIT License)
    --------------------------------------------------------------------------------
    Copyright (c) 2005 - 2014 G-Truc Creation
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    Restrictions:
     By making use of the Software for military purposes, you choose to make a
     Bunny unhappy.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    ================================================================================
    The MIT License
    --------------------------------------------------------------------------------
    Copyright (c) 2005 - 2014 G-Truc Creation
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
     */

    T Coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    T Coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
    T Coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

    T Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    T Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
    T Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

    T Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    T Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
    T Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

    T Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    T Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
    T Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

    T Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    T Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
    T Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

    T Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
    T Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
    T Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

    using Vector = ArrayND<4, T, ISE>;

    Vector Fac0(Coef00, Coef00, Coef02, Coef03);
    Vector Fac1(Coef04, Coef04, Coef06, Coef07);
    Vector Fac2(Coef08, Coef08, Coef10, Coef11);
    Vector Fac3(Coef12, Coef12, Coef14, Coef15);
    Vector Fac4(Coef16, Coef16, Coef18, Coef19);
    Vector Fac5(Coef20, Coef20, Coef22, Coef23);

    Vector Vec0(m[1][0], m[0][0], m[0][0], m[0][0]);
    Vector Vec1(m[1][1], m[0][1], m[0][1], m[0][1]);
    Vector Vec2(m[1][2], m[0][2], m[0][2], m[0][2]);
    Vector Vec3(m[1][3], m[0][3], m[0][3], m[0][3]);

    Vector Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
    Vector Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
    Vector Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
    Vector Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

    Vector SignA(+1, -1, +1, -1);
    Vector SignB(-1, +1, -1, +1);
    MatrixND<4, 4, T, ISE> Inverse{ Inv0 * SignA, Inv1 * SignB, Inv2 * SignA,
        Inv3 * SignB };

    Vector Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

    Vector Dot0(m[0] * Row0);
    T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

    T OneOverDeterminant = static_cast<T>(1) / Dot1;

    return Inverse * OneOverDeterminant;
}

NAMESPACE_PIMATH_END