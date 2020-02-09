// =============================================================
// PI-MATH Mathematics Functions
// Some useful functions
// =============================================================
#pragma once
#include "utils.h"
#include "../pimath.h"

PIMATH_NAMESPACE_BEGIN

/// clamp
/// Input: data a, minimum tmin, maximum tmax
/// Output: a within [tmin, tmax]
template<typename T>
T clamp(const T& a, T min, T max)
{
	if (min > max) return a;
	if (a < min) a = min;
	if (a > max) a = max;
	return a;
}

/// There is a matrix version in matrix.h

/// lerp
/// Input: data a, data b, time t [0, 1]
/// Output: (1 - t) * a + t * b
template<typename T>
T lerp(const T& a, const T& b, real t)
{
	if (t < 0_f) t = 0_f;
	if (t > 1_f) t = 1_f;
	return (1 - t) * a + t * b;
}

PIMATH_NAMESPACE_END