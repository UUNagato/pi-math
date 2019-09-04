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

NAMESPACE_PIMATH_BEGIN

// For N dimensional vector
template<int dim, typename T, InstSetExt ISE = default_instruction_set, typename Enable = void>
struct VectorBase 
{
    T data[dim];
};

// Special form for 1, 2, 3, 4 dimensions.
template<typename T, InstSetExt ISE>
struct VectorBase<1, T, ISE>
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
struct VectorBase<2, T, ISE>
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
struct VectorBase<3, T, ISE, 
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
    VectorBase<3, float32, ISE,
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

    VectorBase(__m128 v) : v(v) {}
    VectorBase(float32 x = 0.0f) : v(_mm_set_ps1(x)) {}
};

template<typename T, InstSetExt ISE>
struct VectorBase<4, T, ISE>
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
    VectorBase<4, float32, ISE,
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

    VectorBase(__m128 v) : v(v) {}
    VectorBase(float32 x = 0.0f) : v(_mm_set_ps1(x)) {}
};

//=================================================================================
// Vectors for both math calculations and graphics (mainly designed for OpenGL)
//=================================================================================
template<int dim, typename T, InstSetExt ISE = default_instruction_set>
struct VectorND : public VectorBase<dim, T, ISE> 
{
    // some special values to determine whether or not using SIMD (from Taichi)
    template<int dim_, typename T_, InstSetExt ISE_>
    static constexpr bool SIMD_FLAG = (dim_ == 3 || dim_ == 4) &&
        std::is_same<T_, float32>::value && ISE_ >= InstSetExt::SSE;

    // constructors =============================================
    VectorND() { 
        for (int i = 0; i < dim; ++i)
            this->data[i] = T();
    }
    VectorND(T val) { 
        for (int i = 0; i < dim; ++i)
            this->data[i] = val; 
    }

    explicit VectorND(const std::array<T, dim> &o) {
        for (int i = 0; i < dim; ++i) {
            this->data[i] = o[i];
        }
    }
    
    // copy constructors
    template <int dim_>
    explicit VectorND(const VectorND<dim_, T, ISE> &o) : VectorND() {
        int dimmin = std::min(dim, dim_);
        for (int i = 0; i < dimmin; ++i)
            this->data[i] = o[i];
    }

    // multi parameters constructor
    explicit VectorND(const std::initializer_list<T> &v) : VectorND() {
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
        VectorND(__m128 value) {
        this->v = value;
    }

    // functions constructors
    template<typename F, std::enable_if_t<std::is_convertible<F, std::function<T(int)>>::value, int> = 0>
    explicit VectorND(const F &f) {
        for (int i = 0; i < dim; ++i)
            this->data[i] = f(i);
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
    PM_INLINE VectorND operator+(const VectorND &v2) const {
        return VectorND([=](int i) { return this->data[i] + v2[i]; });
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE VectorND operator-(const VectorND &v2) const {
        return VectorND([=](int i) { return this->data[i] - v2[i]; });
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE VectorND operator*(const VectorND &v2) const {
        return VectorND([=](int i) { return this->data[i] * v2[i]; });
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<!SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE VectorND operator/(const VectorND &v2) const {
        return VectorND([=](int i) { return this->data[i] / v2[i]; });
    }

    PM_INLINE VectorND& operator=(const VectorND &v2) {
        memcpy(this, &v2, sizeof(*this));
        return *this;
    }

    PM_INLINE bool operator==(const VectorND &v2) const {
        for (int i = 0; i < dim; ++i)
            if (!(this->data[i] == v2[i]))
                return false;
        return true;
    }

    // SIMD ones
    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE VectorND operator+(const VectorND &v2) const {
        return VectorND(_mm_add_ps(this->v, v2.v));
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE VectorND operator-(const VectorND &v2) const {
        return VectorND(_mm_sub_ps(this->v, v2.v));
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE VectorND operator*(const VectorND &v2) const {
        return VectorND(_mm_mul_ps(this->v, v2.v));
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE VectorND operator/(const VectorND &v2) const {
        return VectorND(_mm_div_ps(this->v, v2.v));
    }

    template<int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE,
        typename std::enable_if_t<SIMD_FLAG<dim_, T_, ISE_>, int> = 0>
        PM_INLINE VectorND& operator=(__m128 v) const {
        this->v = v;
        return *this;
    }

    // some other vector operators
    T dot(const VectorND &v2) const {
        T ret = T();
        for (int i = 0; i < dim; ++i)
            ret += this->data[i] * v2[i];
        return ret;
    }

    template<typename std::enable_if_t<dim == 3, int> = 0>
    VectorND cross(const VectorND &v2) const {
        VectorND ret;
        ret[0] = this->y * v2.z - this->z * v2.y;
        ret[1] = this->z * v2.x - this->x * v2.z;
        ret[2] = this->x * v2.y - this->y * v2.x;
        return ret;
    }
};

// IO for vectors
template<int dim, typename T, InstSetExt ISE>
const std::ostream& operator<<(std::ostream &os, const VectorND<dim, T, ISE>&v)
{
    for (int i = 0; i < dim - 1; ++i)
        os << v.data[i] << ',';
    if (dim - 1 >= 0)
        os << v.data[dim - 1];
    return os;
}

NAMESPACE_PIMATH_END