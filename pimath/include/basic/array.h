#pragma once
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <functional>
#include <type_traits>
#include <immintrin.h>
#include <initializer_list>

#include "utils.h"
#include "datatype.h"

PIMATH_NAMESPACE_BEGIN

// For N dimensional vector
template<int dim, typename T, InstSetExt ISE = default_instruction_set, typename Enable = void>
struct ArrayBase 
{
    T data[dim];
};

// Special form for 1, 2, 3, 4 dimensions.
template<typename T, InstSetExt ISE>
struct ArrayBase<1, T, ISE>
{
    union {
        T data[1];
        struct {
            T x;
        };
        struct {
            T r;
        };
    };
};

template<typename T, InstSetExt ISE>
struct ArrayBase<2, T, ISE>
{
    union {
        T data[2];
        struct {
            T x, y;
        };
        struct {
            T r, g;
        };
    };
};

template<typename T, InstSetExt ISE>
struct ArrayBase<3, T, ISE,
    typename std::enable_if_t<(!std::is_same<T, float32>::value || ISE < InstSetExt::SSE)>>
{
    union {
        T data[3];
        struct {
            T x, y, z;
        };
        struct {
            T r, g, b;
        };
    };
};

// SIMD one
template<InstSetExt ISE>
struct PM_ALIGNED(16)
	ArrayBase<3, float32, ISE,
    typename std::enable_if_t<ISE >= InstSetExt::SSE>>
{
    union {
        __m128 v;
        float32 data[4];
        struct {
            float32 x, y, z;
        };
        struct {
            float32 r, g, b;
        };
    };

	ArrayBase(__m128 v) : v(v) {}
	ArrayBase(float32 x = 0.0f) : v(_mm_set_ps1(x)) {}
};

template<InstSetExt ISE>
struct PM_ALIGNED(16)
    ArrayBase<3, float64, ISE,
    typename std::enable_if_t<ISE >= InstSetExt::AVX>>
{
    union {
        __m256d v;
        float64 data[4];
        struct {
            float64 x, y, z;
        };
        struct {
            float64 r, g, b;
        };
    };

    ArrayBase(__m256d _v) : v(_v) {}
    ArrayBase(float64 x = 0.0f) : v(_mm256_set1_pd(x)) {}
};

template<typename T, InstSetExt ISE>
struct ArrayBase<4, T, ISE,
	typename std::enable_if_t<(!std::is_same<T, float32>::value || ISE < InstSetExt::SSE)>>
{
    union {
        T data[4];
        struct {
            T x, y, z, w;
        };
        struct {
            T r, g, b, a;
        };
    };
};

// SIMD one
template<InstSetExt ISE>
struct PM_ALIGNED(16)
    ArrayBase<4, float32, ISE,
    typename std::enable_if_t<ISE >= InstSetExt::SSE>>
{
    union {
        __m128 v;
        float32 data[4];
        struct {
            float32 x, y, z, w;
        };
        struct {
            float32 r, g, b, a;
        };
    };

    ArrayBase(__m128 v) : v(v) {}
    ArrayBase(float32 x = 0.0f) : v(_mm_set_ps1(x)) {}
};

//=================================================================================
// Vectors for both math calculations and graphics (mainly designed for OpenGL)
//=================================================================================
template<int dim, typename T, InstSetExt ISE = default_instruction_set>
struct ArrayND : public ArrayBase<dim, T, ISE> 
{
    // some special values to determine whether or not using SIMD (from Taichi)
    template<int dim_, typename T_, InstSetExt ISE_>
    static constexpr bool SIMD_FLAG = (dim_ == 3 || dim_ == 4) &&
        std::is_same<T_, float32>::value && ISE_ >= InstSetExt::SSE;

    // constructors =============================================
    ArrayND() { 
        for (int i = 0; i < dim; ++i)
            this->data[i] = T();
    }
    explicit ArrayND(T val) { 
        for (int i = 0; i < dim; ++i)
            this->data[i] = val; 
    }

    explicit ArrayND(const std::array<T, dim> &o) {
        for (int i = 0; i < dim; ++i) {
            this->data[i] = o[i];
        }
    }
    
    // copy constructors
    template <int dim_>
    explicit ArrayND(const ArrayND<dim_, T, ISE> &o) : ArrayND() {
        int dimmin = std::min(dim, dim_);
        for (int i = 0; i < dimmin; ++i)
            this->data[i] = o[i];
    }

    // multi parameters constructor
    explicit ArrayND(const std::initializer_list<T> &v) : ArrayND() {
        int i = 0;
        for (auto iter : v) {
            if (i >= dim)
                break;
            this->data[i] = iter;
            ++i;
        }
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        ArrayND(__m128 value) {
        this->v = value;
    }

    // functions constructors
    template<typename F, std::enable_if_t<std::is_convertible<F, std::function<T(int)>>::value, int> = 0>
    explicit ArrayND(const F &f) {
        for (int i = 0; i < dim; ++i)
            this->data[i] = f(i);
    }

    // special case constructors
    explicit ArrayND(T v1, T v2) {
        this->data[0] = v1;
        this->data[1] = v2;
    }

    explicit ArrayND(T v1, T v2, T v3) {
        this->data[0] = v1;
        this->data[1] = v2;
        this->data[2] = v3;
    }

    explicit ArrayND(T v1, T v2, T v3, T v4) {
        this->data[0] = v1;
        this->data[1] = v2;
        this->data[2] = v3;
        this->data[3] = v4;
    }
    
    // =========================================================

    T& operator[](size_t index) {
        return this->data[index];
    }

    const T operator[](size_t index) const {
        return this->data[index];
    }

    T& operator()(size_t index) {
        return this->data[index];
    }

    const T operator()(size_t index) const {
        return this->data[index];
    }

    // Arithmetics
    // General ones
    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE, 
        typename std::enable_if_t<!SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
    PM_INLINE ArrayND operator+(const ArrayND &v2) const {
        return ArrayND([=](int i) { return this->data[i] + v2[i]; });
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE ArrayND operator-(const ArrayND &v2) const {
        return ArrayND([=](int i) { return this->data[i] - v2[i]; });
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE ArrayND operator*(const ArrayND &v2) const {
        return ArrayND([=](int i) { return this->data[i] * v2[i]; });
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE ArrayND operator/(const ArrayND &v2) const {
        return ArrayND([=](int i) { return this->data[i] / v2[i]; });
    }

    PM_INLINE ArrayND& operator=(const ArrayND &v2) {
        memcpy(this, &v2, sizeof(*this));
        return *this;
    }

    PM_INLINE bool operator==(const ArrayND &v2) const {
        for (int i = 0; i < dim; ++i)
            if (!(this->data[i] == v2[i]))
                return false;
        return true;
    }

    PM_INLINE bool operator!=(const ArrayND& v2) const {
        return !((*this) == v2);
    }

    // SIMD ones
    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE ArrayND operator+(const ArrayND &v2) const {
        return ArrayND(_mm_add_ps(this->v, v2.v));
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE ArrayND operator-(const ArrayND &v2) const {
        return ArrayND(_mm_sub_ps(this->v, v2.v));
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE ArrayND operator*(const ArrayND &v2) const {
        return ArrayND(_mm_mul_ps(this->v, v2.v));
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE ArrayND operator/(const ArrayND &v2) const {
        return ArrayND(_mm_div_ps(this->v, v2.v));
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE ArrayND& operator=(__m128 v) const {
        this->v = v;
        return *this;
    }

	// ========================================================================
	// scalar arithmetic
	// ========================================================================
	PM_INLINE ArrayND operator+(const T scalar) const {
		if (dim > 4)
			return ArrayND([=](int i) { return this->data[i] + scalar; });
		else
			return (*this) + ArrayND(scalar);			// using this way to leverage SIMD
	}
	PM_INLINE ArrayND& operator+=(const T scalar) {
		if (dim > 4)
			for (int i = 0; i < dim; ++i)
				this->data[i] = this->data[i] + scalar;
		else
			*this = (*this) + ArrayND(scalar);
		return (*this);
	}

	PM_INLINE ArrayND operator-(const T scalar) const {
		if (dim > 4)
			return ArrayND([=](int i) { return this->data[i] - scalar; });
		else
			return (*this) - ArrayND(scalar);
	}
	PM_INLINE ArrayND& operator-=(const T scalar) {
		if (dim > 4)
			for (int i = 0; i < dim; ++i)
				this->data[i] = this->data[i] - scalar;
		else
			*this = (*this) - ArrayND(scalar);
		return (*this);
	}
	PM_INLINE ArrayND operator*(const T scalar) const {
		if (dim > 4)
			return ArrayND([=](int i) { return this->data[i] * scalar; });
		else
			return (*this) * ArrayND(scalar);
	}
	PM_INLINE ArrayND& operator*=(const T scalar) {
		if (dim > 4)
			for (int i = 0; i < dim; ++i)
				this->data[i] = this->data[i] * scalar;
		else
			*this = (*this) * ArrayND(scalar);
		return (*this);
	}
	PM_INLINE ArrayND operator/(const T scalar) const {
		if (dim > 4)
			return ArrayND([=](int i) { return this->data[i] / scalar; });
		else
			return (*this) / ArrayND(scalar);
	}
	PM_INLINE ArrayND& operator/=(const T scalar) {
		if (dim > 4)
			for (int i = 0; i < dim; ++i)
				this->data[i] = this->data[i] / scalar;
		else
			*this = (*this) / ArrayND(scalar);
		return (*this);
	}

    // Sqrt
    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
    static ArrayND sqrt(const ArrayND &a) {
        return ArrayND([=](int i) { return std::sqrt(a.data[i]); });
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
    static ArrayND sqrt(const ArrayND &a) {
        return ArrayND(_mm_sqrt_ps(a.v));
    }

    // some other vector operators
    T dot(const ArrayND &v2) const {
        T ret = T();
        for (int i = 0; i < dim; ++i)
            ret += this->data[i] * v2[i];
        return ret;
    }

    // NaNs
    bool hasNaNs() const {
        bool n = false;
        for (int i = 0; i < dim; ++i)
            n = n || std::isnan(this->data[i]);
        return n;
    }
};

// IO for vectors
template<int dim, typename T, InstSetExt ISE>
const std::ostream& operator<<(std::ostream &os, const ArrayND<dim, T, ISE>&v)
{
    for (int i = 0; i < dim - 1; ++i)
        os << v.data[i] << ',';
    if (dim - 1 >= 0)
        os << v.data[dim - 1];
    return os;
}

PIMATH_NAMESPACE_END