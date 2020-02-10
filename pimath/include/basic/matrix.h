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

PIMATH_NAMESPACE_BEGIN

template<int rows, int cols, typename T, InstSetExt ISE>
struct MatrixBase
{
    ArrayND<rows, T, ISE> data[cols];
};

// special accessor for Vector-like Matrix
template<typename T, InstSetExt ISE>
struct MatrixBase<1, 1, T, ISE>
{
    MatrixBase() : data() {}
    union {
        ArrayND<1, T, ISE> data[1];
        struct {
            T x;
        };
        struct {
            T r;
        };
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<2, 1, T, ISE>
{
    MatrixBase() : data() {}
    union {
		ArrayND<2, T, ISE> data[1];
        struct {
            T x, y;
        };
        struct {
            T r, g;
        };
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<3, 1, T, ISE>
{
    MatrixBase() : data() {}
    union {
		ArrayND<3, T, ISE> data[1];
        struct {
            T x, y, z;
        };
        struct {
            T r, g, b;
        };
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<4, 1, T, ISE>
{
    MatrixBase() : data() {}
    union {
		ArrayND<4, T, ISE> data[1];
        struct {
            T x, y, z, w;
        };
        struct {
            T r, g, b, a;
        };
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 2, T, ISE>
{
    MatrixBase() : data() {}
    union {
		ArrayND<1, T, ISE> data[2];
        struct {
            T x, y;
        };
        struct {
            T r, g;
        };
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 3, T, ISE>
{
    MatrixBase() : data() {}
    union {
		ArrayND<1, T, ISE> data[3];
        struct {
            T x, y, z;
        };
        struct {
            T r, g, b;
        };
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 4, T, ISE>
{
    MatrixBase() : data() {}
    union {
		ArrayND<1, T, ISE> data[4];
        struct {
            T x, y, z, w;
        };
        struct {
            T r, g, b, a;
        };
    };
};

template<int rows, int cols, typename T, InstSetExt ISE = default_instruction_set>
struct MatrixND : public MatrixBase<rows, cols, T, ISE>
{
    template<int rows_ = rows, int cols_ = cols>
    static constexpr bool IS_VECTOR = (cols_ == 1) || (rows_ == 1);

    template<int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    static constexpr bool MATRIX_SSE = ((cols_ == 3) || (cols_ == 4)) && std::is_same<T_, float32>::value && ISE_ >= InstSetExt::SSE;

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

    // Special Vector Constructor
    template<int rows_ = rows, int cols_ = cols,
        int vrows = rows, int vcols = cols, InstSetExt ISE_ = ISE,
    typename std::enable_if_t<IS_VECTOR<rows_, cols_> && IS_VECTOR<vrows, vcols>, int> = 0>
        MatrixND(const MatrixND<vrows, vcols, T, ISE_>& v, T default_value) {
        int size_a = std::max(rows_, cols_);
        int size_b = std::max(vrows, vcols);
        int shorter = std::min(size_a, size_b);
        int longer = std::max(size_a, size_b);
        for (int i = 0; i < shorter; ++i)
            (*this)[i] = v[i];
        for (int i = shorter; i < longer; ++i)
            (*this)[i] = default_value;
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

    explicit MatrixND(T x, T y, T z) {
        static_assert((rows == 1 && cols == 3) || (rows == 3 && cols == 1), "Matrix must be a 3-dimensional vector");
        this->x = x;
        this->y = y;
        this->z = z;
    }

    explicit MatrixND(T x, T y, T z, T w) {
        static_assert((rows == 1 && cols == 4) || (rows == 4 && cols == 1), "Matrix must be a 3-dimensional vector");
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
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
    template<int rows_ = rows, int cols_ = cols,
        typename std::enable_if_t<IS_VECTOR<rows_, cols_>, int> = 0>
        PM_INLINE T& operator[] (size_t index) {
        if (cols_ == 1)
            return this->data[0][index];
        else
            return this->data[index][0];
    }

    template<int rows_ = rows, int cols_ = cols,
        typename std::enable_if_t<!IS_VECTOR<rows_, cols_>, int> = 0>
    PM_INLINE ArrayND<rows, T, ISE>& operator[] (size_t index) {
        return this->data[index];
    }

    template<int rows_ = rows, int cols_ = cols,
        typename std::enable_if_t<IS_VECTOR<rows_, cols_>, int> = 0>
        PM_INLINE const T& operator[] (size_t index) const {
        if (cols_ == 1)
            return this->data[0][index];
        else
            return this->data[index][0];
    }

    template<int rows_ = rows, int cols_ = cols,
        typename std::enable_if_t<!IS_VECTOR<rows_, cols_>, int> = 0>
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

    PM_INLINE bool operator== (const MatrixND& m) const {
        for (int i = 0; i < cols; ++i)
            if (!(this->data[i] == m.data[i]))
                return false;
        return true;
    }

    PM_INLINE bool operator!= (const MatrixND& m) const {
        for (int i = 0; i < cols; ++i)
            if (this->data[i] != m.data[i])
                return true;
        return false;
    }

    // Arithmatic
    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND operator+ (const MatrixND& m) const {
        return MatrixND([=](int c) { return this->data[c] + m.data[c]; });
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND& operator+= (const MatrixND& m) {
        for (int c = 0; c < cols; ++c)
            this->data[c] = this->data[c] + m.data[c];
        return *this;
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND operator- (const MatrixND& m) const {
        return MatrixND([=](int c) { return this->data[c] - m.data[c]; });
    }

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND& operator-= (const MatrixND& m) {
        for (int c = 0; c < cols; ++c)
            this->data[c] = this->data[c] - m.data[c];
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
        // pack first matrix rows into four __m128
        float r[4][4] = { 0 };
        for (int i = 0; i < lrows; ++i)
            for (int j = 0; j < lcols; ++j)
                r[i][j] = m1(i, j);
        __m128 rr[4];
        rr[0] = _mm_load_ps(r[0]);
        rr[1] = _mm_load_ps(r[1]);
        rr[2] = _mm_load_ps(r[2]);
        rr[3] = _mm_load_ps(r[3]);

        for (int r = 0; r < lrows; ++r) {
            for (int c = 0; c < rcols; ++c) {
                _mm_store_ss(&(ret->operator()(r, c)), _mm_dp_ps(rr[r], m2.data[c].v, 0xf1));
            }
        }
    }

    template<int lrows = rows, int lcols = cols, int rrows, int rcols,
            typename T_ = T, InstSetExt ISE_ = ISE,
            typename std::enable_if_t<MATRIX_SSE<lcols, T, ISE> && MATRIX_SSE<rrows, T_, ISE_> &&
                                    (!IS_VECTOR<lrows, lcols> || !IS_VECTOR<rrows, rcols>), int> = 0>
    PM_INLINE MatrixND<lrows, rcols, T, ISE> operator*(const MatrixND<rrows, rcols, T_, ISE_>& m2) const {
        static_assert(lcols == rrows, "The multiplication matrix have incompatible dimensions.");
        MatrixND<rows, rcols, T, ISE> ret;
        multiply_SSE(*this, m2, &ret);
        return ret;
    }

    template<int lrows = rows, int lcols = cols, int rrows, int rcols,
        typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<MATRIX_SSE<lcols, T, ISE> && MATRIX_SSE<rrows, T_, ISE_> &&
                                    (!IS_VECTOR<lrows, lcols> || !IS_VECTOR<rrows, rcols>), int> = 0>
        PM_INLINE MatrixND<lrows, rcols, T, ISE>& operator*=(const MatrixND<rrows, rcols, T_, ISE_>& m2) {
        static_assert(lcols == rrows, "The multiplication matrix have incompatible dimensions.");
        multiply_SSE(*this, m2, this);
        return *this;
    }

    template<int lrows = rows, int lcols = cols, int rrows, int rcols,
                typename T_ = T, InstSetExt ISE_ = ISE,
                typename std::enable_if_t<!MATRIX_SSE<lcols, T, ISE> || !MATRIX_SSE<rrows, T_, ISE_> &&
                        (!IS_VECTOR<lrows, lcols> || !IS_VECTOR<rrows, rcols>), int> = 0>
        PM_INLINE MatrixND<lrows, rcols, T, ISE> operator*(const MatrixND<rrows, rcols, T_, ISE_>& m2) const {
        static_assert(lcols == rrows, "The multiplication matrix have incompatible dimensions.");
        MatrixND<rows, rcols, T, ISE> ret;
        multiply(*this, m2, &ret);
        return ret;
    }

    template<int lrows = rows, int lcols = cols, int rrows, int rcols,
                typename T_ = T, InstSetExt ISE_ = ISE,
                typename std::enable_if_t<!MATRIX_SSE<lcols, T, ISE> || !MATRIX_SSE<rrows, T_, ISE_> &&
                        (!IS_VECTOR<lrows, lcols> || !IS_VECTOR<rrows, rcols>), int> = 0>
        PM_INLINE MatrixND<lrows, rcols, T, ISE>& operator*=(const MatrixND<rrows, rcols, T_, ISE_>& m2) {
        static_assert(lcols == rrows, "The multiplication matrix have incompatible dimensions.");
        multiply(*this, m2, this);
        return *this;
    }

    PM_INLINE MatrixND operator-()
    {
        return MatrixND([=](int i) { return -this->data[i]; });
    }

    // A really special function used for speed up transformation, directly transform a vector3 by a 4x4 matrix
    template<typename T_ = T, InstSetExt ISE_ = ISE,
            typename std::enable_if_t<MATRIX_SSE<4, T_, ISE_> && MATRIX_SSE<3, T_, ISE_>, int> = 0>
    PM_INLINE MatrixND<3, 1, T, ISE> applyTransform(const MatrixND<3, 1, T, ISE>& m2)
    {
        static_assert(rows == 4 && cols == 4, "applyTransform can only be used for 4x4 matrix times 3-length colume vector");
        MatrixND<3, 1, T, ISE> ret;
        // pack first matrix rows into four __m128
        float r[3][4] = { 0 };
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                r[i][j] = (*this)(i, j);
        __m128 rr[3];
        rr[0] = _mm_load_ps(r[0]);
        rr[1] = _mm_load_ps(r[1]);
        rr[2] = _mm_load_ps(r[2]);

        for (int r = 0; r < 3; ++r) {
            _mm_store_ss(&ret(r, 0), _mm_dp_ps(rr[r], m2.data[0].v, 0xf1));
        }
        return ret;
    }

    template<typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!(MATRIX_SSE<4, T_, ISE_> && MATRIX_SSE<3, T_, ISE_>), int> = 0>
        PM_INLINE MatrixND<3, 1, T, ISE> applyTransform(const MatrixND<3, 1, T, ISE>& m2)
    {
        static_assert(rows == 4 && cols == 4, "applyTransform can only be used for 4x4 matrix times 3-length colume vector");
        MatrixND<3, 1, T, ISE> ret;
        for (int r = 0; r < 3; ++r) {
            T v = T(0);
            for (int c = 0; c < 3; ++c)
                v = v + (*this)(r, c) * m2(c, 0);
            ret(r, 0) = v;
        }

        return ret;
    }

    // ================================================================================
    // Multiplication between two vectors.
    // Element-wise
    // ================================================================================
private:
    template<int rows_ = rows, int cols_ = cols, int rrows, int rcols, typename T_ = T,
                InstSetExt ISE_ = ISE>
    void elementwise_multiply(const MatrixND<rows_, cols_, T_, ISE_> &v1,
                                const MatrixND<rrows, rcols, T_, ISE_> &v2,
                                MatrixND<rows_, cols_, T_, ISE_> *ret) {
        static_assert(rows_ == rrows && cols_ == rcols);
        if (!ret)
            return;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                ret(i, j) = v1(i, j) * v2(i, j);
            }
        }
    }

public:
    template<int rows_ = rows, int cols_ = cols,
        typename std::enable_if_t<IS_VECTOR<rows_, cols_>, int> = 0>
        MatrixND operator* (const MatrixND<rows, cols, T, ISE>& v2) const {
        MatrixND ret;
        for (int i = 0; i < cols; ++i)
            ret.data[i] = this->data[i] * v2.data[i];
        return ret;
    }

    template<int rows_ = rows, int cols_ = cols,
        typename std::enable_if_t<IS_VECTOR<rows_, cols_>, int> = 0>
        MatrixND& operator*= (const MatrixND<rows, cols, T, ISE>& v2) {
        for (int i = 0; i < cols; ++i)
            this->data[i] = this->data[i] * v2.data[i];
        return *this;
    }

    // ================================================================================
    // Scalar
    // ================================================================================
    PM_INLINE MatrixND operator+ (const T scalar) const {
        return MatrixND([=](int c) { return this->data[c] + scalar; });
    }

    PM_INLINE MatrixND operator- (const T scalar) const {
        return MatrixND([=](int c) { return this->data[c] - scalar; });
    }
    
    PM_INLINE MatrixND operator* (const T scalar) const {
        return MatrixND([=](int c) { return this->data[c] * scalar; });
    }
    /*
    friend PM_INLINE MatrixND operator* (const T scalar, const MatrixND& m) {
        return MatrixND([=](int c) { return m.data[c] * scalar; });
    }*/

    PM_INLINE MatrixND operator/ (const T scalar) const {
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
    PM_INLINE MatrixND<cols, rows, T, ISE> transpose() const {
        return MatrixND<cols, rows, T, ISE>([=](int r, int c) { return (*this)(c, r); });
    }

    friend PM_INLINE MatrixND transpose(const MatrixND& m) {
        return m.transpose();
    }

    PM_INLINE T determinant() {
        static_assert(rows == cols, "Determinant can only be calculated on squared matrix");

    }

	// ===============================================================================
	// Operations for vectors
	// ===============================================================================
	PM_INLINE T length2() const {
		T total_sqr = T(0);
		for (int c = 0; c < cols; ++c)
			for (int r = 0; r < rows; ++r)
				total_sqr += (*this)(r, c) * (*this)(r, c);
		return total_sqr;
	}

	PM_INLINE T length() const {
		T sqrLen = length2();
		return sqrt(sqrLen);
	}

    friend PM_INLINE T distance(const MatrixND& p1, const MatrixND& p2) {
        return (p1 - p2).length();
    }

	PM_INLINE MatrixND normalize() const {
        T len = length();
        if (len == T(0))
            return MatrixND(T(0));
        else
            return (*this) / len;
	}

    PM_INLINE MatrixND& normalizeInPlace() {
        T len = length();
        if (len != T(0))
            (*this) /= len;
        return *this;
    }

    friend PM_INLINE MatrixND normalize(const MatrixND& v) {
        return v.normalize();
    }

    // Dot
    PM_INLINE T dot(const MatrixND& q) const {
        T sum = T(0);
        for (int i = 0; i < cols; ++i) {
            sum = sum + this->data[i].dot(q.data[i]);
        }
        return sum;
    }

    friend PM_INLINE T dot(const MatrixND& m1, const MatrixND& m2) {
        return m1.dot(m2);
    }

    // NaNs check
    bool hasNaNs() const {
        bool n = false;
        for (int i = 0; i < cols; ++i)
            n = n || this->data[cols].hasNaNs();
        return n;
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
// Cross
//======================================================================
template<int rows, int cols, int rows_, int cols_, typename T, InstSetExt ISE, InstSetExt ISE_>
MatrixND<rows, cols, T, ISE> cross(const MatrixND<rows, cols, T, ISE>& v1, const MatrixND<rows_, cols_, T, ISE_>& v2) {
    static_assert((rows == 3 && cols == 1) || (rows == 1 && cols == 3), "cross product only works for 3 dimension vectors");
    static_assert((rows_ == 3 && cols_ == 1) || (rows_ == 1 && cols_ == 3), "cross product only works for 3 dimension vectors");
    MatrixND<3, 1, T, ISE> ret;
    ret.x = v1.y * v2.z - v1.z * v2.y;
    ret.y = v1.z * v2.x - v1.x * v2.z;
    ret.z = v1.x * v2.y - v1.y * v2.x;
    return ret;
}

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
MatrixND<4, 4, T, ISE> inverse(const MatrixND<4, 4, T, ISE>& m) {
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

PIMATH_NAMESPACE_END