// ========================================================
// PI-Math, Quaternion
// ========================================================
#pragma once

#include "utils.h"
#include "matrix.h"

PIMATH_NAMESPACE_BEGIN

template<InstSetExt ISE = default_instruction_set>
struct QuaternionBase
{
	using RealPart = MatrixND<3, 1, real, ISE>;

public:
	QuaternionBase() : v(0_f), w(1_f) {}
	QuaternionBase(RealPart r, real w) : v(r), w(w) {}
	QuaternionBase(real x, real y, real z, real w) : v(x, y, z), w(w) {}

	QuaternionBase(const QuaternionBase& q) : v(q.v), w(q.w) {}

	//=============================================================
	// Operators - copy and logic
	//=============================================================
	PM_INLINE QuaternionBase& operator= (const QuaternionBase& q) {
		v = q.v;
		w = q.w;
		return *this;
	}

	PM_INLINE bool operator== (const QuaternionBase& q) const {
		return (v == q.v && w == q.w);
	}

	PM_INLINE bool operator!= (const QuaternionBase& q) const {
		return !(this->operator==(q));
	}

	//=============================================================
	// Operators - Arithmetics
	//=============================================================
	PM_INLINE QuaternionBase operator+ (const QuaternionBase& q) const {
		return QuaternionBase(v + q.v, w + q.w);
	}

	PM_INLINE QuaternionBase& operator+= (const QuaternionBase& q) {
		v = v + q.v;
		w = w + q.w;
		return *this;
	}

	PM_INLINE QuaternionBase operator- (const QuaternionBase& q) const {
		return QuaternionBase(v - q.v, w - q.w);
	}
	
	PM_INLINE QuaternionBase& operator-= (const QuaternionBase& q) {
		v = v - q.v;
		w = w - q.w;
		return *this;
	}

	PM_INLINE QuaternionBase operator* (const real s) const {
		return QuaternionBase(v * s, w * s);
	}

	PM_INLINE QuaternionBase operator*= (const real s) {
		v = v * s;	w = w * s;
		return *this;
	}

	PM_INLINE QuaternionBase operator/ (const real s) const {
		return QuaternionBase(v / s, w / s);
	}

	PM_INLINE QuaternionBase operator/= (const real s) {
		v = v / s;	w = w / s;
		return *this;
	}

	PM_INLINE QuaternionBase operator* (const QuaternionBase& q) const {
		QuaternionBase ret;
		ret.v = cross(v, q.v) + q.w * v + w * q.v;
		ret.w = w * q.w - dot(v, q.v);
		return ret;
	}
	
	// =======================================================
	// Other operations
	// =======================================================
	PM_INLINE real dot(const QuaternionBase& q) const {
		real rp = v.dot(q.v);
		real wp = w * q.w;
		return rp + wp;
	}

	friend PM_INLINE real dot(const QuaternionBase& p, const QuaternionBase& q) {
		return p.dot(q);
	}

	PM_INLINE QuaternionBase normalize() const {
		real sqrLength = dot(*this);
		return (*this) / std::sqrt(sqrLength);
	}

	friend PM_INLINE QuaternionBase normalize(const QuaternionBase& q) {
		return q.normalize();
	}

	QuaternionBase slerp(real t, const QuaternionBase& q1, const QuaternionBase& q2) {
		real cosTheta = q1.dot(q2);
		if (cosTheta > .9995_f)
			return normalize(((1_f - t) * q1 + t * q2));

		real theta = std::acos(clamp(cosTheta, -1_f, 1_f));
		real thetap = theta * t;
		QuaternionBase qperp = normalize((q2 - q1 * cosTheta));
		return q1 * std::cos(thetap) + qperp * std::sin(thetap);
	}

	MatrixND<3, 3, real, ISE> toMatrix() const {
		MatrixND<3, 3, real, ISE> mat;
		real xy = v.x * v.y, yz = v.y * v.z, xz = v.x * v.z;
		RealPart v2 = v * v;
		RealPart vw = v * w;
		mat(0, 0) = 1_f - 2_f * (v2.y + v2.z);
		mat(0, 1) = 2_f * (xy + vw.z);
		mat(0, 2) = 2_f * (xz - vw.y);
		mat(1, 0) = 2_f * (xy - vw.z);
		mat(1, 1) = 1_f - 2_f * (v2.x + v2.z);
		mat(1, 2) = 2_f * (yz + vw.x);
		mat(2, 0) = 2_f * (xz + vw.y);
		mat(2, 1) = 2_f * (yz - vw.x);
		mat(2, 2) = 1_f - 2_f * (v2.x + v2.y);
		return mat;
	}

public:
	RealPart v;
	real w;
};

using Quaternion = QuaternionBase<default_instruction_set>;

PIMATH_NAMESPACE_END