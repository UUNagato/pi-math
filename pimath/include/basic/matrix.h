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
    union {
        ArrayND<1, T, ISE> data[1];
        T x;
        T r;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<2, 1, T, ISE>
{
    union {
		ArrayND<2, T, ISE> data[1];
        T x, y;
        T r, g;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<3, 1, T, ISE>
{
    union {
		ArrayND<3, T, ISE> data[1];
        T x, y, z;
        T r, g, b;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<4, 1, T, ISE>
{
    union {
		ArrayND<4, T, ISE> data[1];
        T x, y, z, w;
        T r, g, b, a;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 2, T, ISE>
{
    union {
		ArrayND<1, T, ISE> data[2];
        T x, y;
        T r, g;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 3, T, ISE>
{
    union {
		ArrayND<1, T, ISE> data[3];
        T x, y, z;
        T r, g, b;
    };
};

template<typename T, InstSetExt ISE>
struct MatrixBase<1, 4, T, ISE>
{
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

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    static constexpr bool MATRIX_SSE_FLAG = ((cols_ == 3 || cols_ == 4) && (rows_ == 3 || rows_ == 4)) &&
        (std::is_same<T_, float32>::value && ISE_ == InstSetExt::SSE);
	template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
	static constexpr bool MATRIX_AVX_FLAG = ((cols_ == 3 || cols_ == 4) && (rows_ == 3 || rows_ == 4)) &&
		((std::is_same<T_, float32>::value || std::is_same<T_, float64>::value) && ISE_ == InstSetExt::AVX);
	template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
	static constexpr bool MATRIX_SIMD_FLAG = MATRIX_SSE_FLAG<rows_, cols_, T_, ISE_> ||
		MATRIX_AVX_FLAG<rows_, cols_, T_, ISE_>;

    // Constructors
    MatrixND() {
        // do nothing since ArrayND will initialize all value.
    }

    // Single value Constructor
    MatrixND(T v) {
        for (int i = 0; i < cols; ++i)
            for (int j = 0; j < rows; ++j) {
                this->data[i][j] = v;
            }
    }

    // Copy Constructor
    MatrixND(const MatrixND &v) {
        memcpy(this, &v, sizeof(*this));
    }

    // Function Constructor
    template<typename F, std::enable_if_t<std::is_convertible<F, std::function<T(int, int)>>::value, int> = 0>
    explicit MatrixND(const F &f) {
        for (int i = 0; i < cols; ++i)
            for (int j = 0; j < rows; ++j) {
                this->data[i][j] = f(j, i);
            }
    }

    template<typename F, std::enable_if_t<std::is_convertible<F, std::function<ArrayND<rows,T,ISE>(int)>>::value, int> = 0>
    explicit MatrixND(const F &f) {
        for (int i = 0; i < cols; ++i)
            this->data[i] = f(i);
    }

    // Variant Value Constructor
    explicit MatrixND(const std::initializer_list<T> &list) : MatrixND() {
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

    PM_INLINE MatrixND& operator= (const MatrixND &m) {
        memcpy(this, &m, sizeof(*this));
        return *this;
    }

    PM_INLINE bool operator== (const MatrixND &m) {
        for (int i = 0; i < cols; ++i)
            if (!(this->data[i] == m[i]))
                return false;
        return true;
    }

    // Arithmatic
    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND operator+ (const MatrixND &m) {
        return MatrixND([=](int c) { return this->data[c] + m[c]; });
    }

	template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
	PM_INLINE MatrixND& operator+= (const MatrixND &m) {
		for (int c = 0; c < cols; ++c)
			this->data[c] = this->data[c] + m[c];
		return *this;
	}

    template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
    PM_INLINE MatrixND operator- (const MatrixND &m) {
        return MatrixND([=](int c) { return this->data[c] - m[c]; });
    }

	template<int rows_ = rows, int cols_ = cols, typename T_ = T, InstSetExt ISE_ = ISE>
	PM_INLINE MatrixND& operator-= (const MatrixND &m) {
		for (int c = 0; c < cols; ++c)
			this->data[c] = this->data[c] - m[c];
		return *this;
	}

	// =======================================================================================
	// Multiplication
	// =======================================================================================
	template<int lrows = rows, int lcols = cols, int rrows = cols, int rcols, typename T_ = T, 
		InstSetExt ISE_ = ISE>
		static void multiply(const MatrixND<lrows, lcols, T_, ISE_> &m1,
			const MatrixND<rrows, rcols, T_, ISE_> &m2, MatrixND<lrows, rcols, T_, ISE_> *ret)
	{
		static_assert(lcols == rrows, "The multiplication matrix have incompatible dimensions.");
		for (int r = 0; r < lrows; ++r) {
			for (int c = 0; c < rcols; ++c) {
				T tmp = T();
				for (int i = 0; i < lcols; ++i) {
					tmp = tmp + this->operator()(r, i) * m(i, c);
				}
				(*ret)(r, c) = tmp;
			}
		}
	}

	template<int lrows = rows, int lcols = cols, int rrows = cols, int rcols, typename T_ = T,
		InstSetExt ISE_ = ISE>
		static void multiply_SSE(const MatrixND<lrows, lcols, T_, ISE_> &m1,
			const MatrixND<rrows, rcols, T_, ISE_> &m2, MatrixND<lrows, rcols, T_, ISE_> *ret)
	{
		static_assert(lcols == rrows, "The multiplication matrix have incompatible dimensions.");
		static_assert(MATRIX_SSE_FLAG<lrows, lcols, T_, ISE_> && MATRIX_SSE_FLAG<rrows, rcols, T_, ISE_>,
			"SSE is not supported");
		
		__m128 r1, r2, r3, r4;

	}

    template<int rrows = cols, int rcols,
        typename std::enable_if_t<!MATRIX_SIMD_FLAG<rows, cols, T, ISE>, int> = 0,
		typename std::enable_if_t<!MATRIX_SIMD_FLAG<rrows, rcols, T, ISE>, int> = 0>
    PM_INLINE MatrixND<rows, rcols, T, ISE> operator* (const MatrixND<rrows, rcols, T, ISE> &m) {
		MatrixND<rows, rcols, T, ISE> ret;
		multiply<rows, cols, rrows, rcols, T, ISE>(*this, m, &ret);
		return ret;
    }

	// SIMD, SSE for matrix multiplication

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
};

// IO for matrix
template<int rows, int cols, typename T, InstSetExt ISE>
const std::ostream& operator<<(std::ostream &os, const MatrixND<rows, cols, T, ISE>&v)
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

NAMESPACE_PIMATH_END