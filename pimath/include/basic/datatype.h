#pragma once
#include "utils.h"

PIMATH_NAMESPACE_BEGIN

// BASIC TYPES
using float32 = float;
using float64 = double;

#if defined(_MSC_VER)
using int64 = __int64;
using int32 = __int32;
using uint64 = unsigned __int64;
using uint32 = unsigned __int32;
#else
using int64 = long long;
using int32 = long;
using uint64 = unsigned long long;
using uint32 = unsigned long;
#endif

#ifdef PIMATH_64BIT
using real = double;
#else
using real = float;
#endif

// from https://github.com/hi2p-perim/lightmetrica-v2
real constexpr operator"" _f(long double v) {
    return real(v);
}

real constexpr operator"" _f(unsigned long long v) {
    return real(v);
}

PIMATH_NAMESPACE_END
