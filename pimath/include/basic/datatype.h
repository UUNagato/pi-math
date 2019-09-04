#pragma once
#include "utils.h"

NAMESPACE_PIMATH_BEGIN

// BASIC TYPES
typedef float float32;
typedef double float64;

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

NAMESPACE_PIMATH_END