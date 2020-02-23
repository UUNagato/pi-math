#pragma once
// =============================================================================
// Functions for sampling
// =============================================================================
#define PIMATH_SAMPLING_BEGIN namespace Sampling {
#define PIMATH_SAMPLING_END }

#include <cmath>
#include "../pimath.h"

PIMATH_NAMESPACE_BEGIN
PIMATH_SAMPLING_BEGIN

template<typename T>
Vector3t<T> uniformSampleHemisphere(const Vector2t<T>& u, T* pdf = nullptr)
{
	T z = u[0];
	T r = std::sqrt(std::max(T(0), T(1) - z * z));
	T phi = T(2) * PI * u[1];
	if (pdf)
		*pdf = INV_2PI;
	return Vector3t<T>(r * std::cos(phi), r * std::sin(phi), z);
}

template<typename T>
Vector3t<T> uniformSampleSphere(const Vector2t<T>& u, T* pdf = nullptr)
{
	T z = T(1) - T(2) * u[0];
	T r = std::sqrt(std::max(T(0), T(1) - z * z));
	T phi = T(2) * PI * u[1];
	if (pdf)
		*pdf = INV_2PI / T(2);
	return Vector3t<T>(r * std::cos(phi), r * std::sin(phi), z);
}

template<typename T>
Vector2t<T> uniformSampleDisk(const Vector2t<T>& u, T* pdf = nullptr)
{
	T r = std::sqrt(u[0]);
	T theta = 2 * PI * u[1];
	if (pdf)
		*pdf = INV_PI;
	return Vector2t<T>(r * std::cos(theta), r * std::sin(theta));
}

template<typename T>
Vector2t<T> concentricSampleDisk(const Vector2t<T>& u, T* pdf = nullptr)
{
	Vector2t<T> uOffset = u * T(2) - Vector2t<T>(T(1));
	if (pdf)
		return INV_PI;

	if (uOffset.x == T(0) && uOffset.y == T(0))
		return Vector2t<T>(T(0));

	T theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = PI / 4_f * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = PI / 2_f - PI / 4_f * (uOffset.x / uOffset.y);
	}
	return r * Vector2t<T>(std::cos(theta), std::sin(theta));
}

template<typename T>
Vector3t<T> cosineSampleHemisphere(const Vector2t<T>& u, T* pdf = nullptr)
{
	Vector2t<T> d = concentricSampleDisk(u);
	T z = std::sqrt(std::max(T(0), T(1) - d.x * d.x - d.y * d.y));
	if (pdf)
		*pdf = z * INV_PI;		// here z is equal to cosTheta
	return Vector3t<T>(d.x, d.y, z);
}
PIMATH_SAMPLING_END
PIMATH_NAMESPACE_END