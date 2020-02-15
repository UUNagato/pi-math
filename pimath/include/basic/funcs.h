#pragma once
// =============================================================
// PI-MATH Mathematics Functions
// Some useful functions
// =============================================================
#include "utils.h"

PIMATH_NAMESPACE_BEGIN

/// clamp
/// Input: data a, minimum tmin, maximum tmax
/// Output: a within [tmin, tmax]
template<typename T>
T clamp(T& a, T min, T max)
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

/// Floor function for Matrix and Vector
template<int rows, int cols, typename T, InstSetExt ISE>
MatrixND<rows, cols, T, ISE> floor(const MatrixND<rows, cols, T, ISE>& m)
{
	MatrixND<rows, cols, T, ISE> ret;
	for (int c = 0; c < cols; ++c)
		for (int r = 0; r < rows; ++r)
			ret(r, c) = std::floor(m(r, c));
	return ret;
}

/// Ceil function for Matrix and Vector
template<int rows, int cols, typename T, InstSetExt ISE>
MatrixND<rows, cols, T, ISE> ceil(const MatrixND<rows, cols, T, ISE>& m)
{
	MatrixND<rows, cols, T, ISE> ret;
	for (int c = 0; c < cols; ++c)
		for (int r = 0; r < rows; ++r)
			ret(r, c) = std::ceil(m(r, c));
	return ret;
}

/// Max function for ArrayND
template<int dim, typename T, InstSetExt ISE>
ArrayND<dim, T, ISE> max(const ArrayND<dim, T, ISE>& v1,
	const ArrayND<dim, T, ISE>& v2)
{
	ArrayND<dim, T, ISE> ret;
	for (int i = 0; i < dim; ++i)
		ret.data[i] = std::max(v1.data[i], v2.data[i]);
	return ret;
}

/// Max function for Matrix and Vector
template<int rows, int cols, typename T, InstSetExt ISE>
MatrixND<rows, cols, T, ISE> max(const MatrixND<rows, cols, T, ISE>& m1,
	const MatrixND<rows, cols, T, ISE>& m2)
{
	MatrixND<rows, cols, T, ISE> ret;
	for (int i = 0; i < cols; ++i)
		ret.data[i] = max(m1.data[i], m2.data[i]);
	return ret;
}

/// Min function for ArrayND
template<int dim, typename T, InstSetExt ISE>
ArrayND<dim, T, ISE> min(const ArrayND<dim, T, ISE>& v1,
	const ArrayND<dim, T, ISE>& v2)
{
	ArrayND<dim, T, ISE> ret;
	for (int i = 0; i < dim; ++i)
		ret.data[i] = std::min(v1.data[i], v2.data[i]);
	return ret;
}

/// Min function for Matrix and Vector
template<int rows, int cols, typename T, InstSetExt ISE>
MatrixND<rows, cols, T, ISE> min(const MatrixND<rows, cols, T, ISE>& m1,
	const MatrixND<rows, cols, T, ISE>& m2)
{
	MatrixND<rows, cols, T, ISE> ret;
	for (int i = 0; i < cols; ++i)
		ret.data[i] = min(m1.data[i], m2.data[i]);
	return ret;
}

/// Radians
/// convert degrees to radians
template<typename T>
T radians(T degrees)
{
	return (degrees / T(180)) * PI;
}

/// Degrees
/// convert radians to degrees
template<typename T>
T degrees(T radians)
{
	return (radians / PI) * T(180);
}



PIMATH_NAMESPACE_END