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

	template<InstSetExt ISE_ = ISE>
	QuaternionBase(const MatrixND<4, 4, real, ISE_>& m) {
		real trace = m(0, 0) + m(1, 1) + m(2, 2);
		if (trace > 0_f) {
			// Compute w from matrix trace, then xyz
			// 4w^2 = m[0][0] + m[1][1] + m[2][2] + m[3][3] (but m[3][3] == 1)
			real s = std::sqrt(trace + 1_f);
			w = s / 2_f;
			s = 0.5_f / s;
			v.x = (m(2, 1) - m(1, 2)) * s;
			v.y = (m(0, 2) - m(2, 0)) * s;
			v.z = (m(1, 0) - m(0, 1)) * s;
		}
		else {
			// Compute largest of $x$, $y$, or $z$, then remaining components
			const int nxt[3] = { 1, 2, 0 };
			real q[3];
			int i = 0;
			if (m(1, 1) > m(0, 0)) i = 1;
			if (m(2, 2) > m(i, i)) i = 2;
			int j = nxt[i];
			int k = nxt[j];
			real s = std::sqrt((m(i, i) - (m(j, j) + m(k, k))) + 1_f);
			q[i] = s * 0.5_f;
			if (s != 0_f) s = 0.5_f / s;
			w = (m(k, j) - m(j, k)) * s;
			q[j] = (m(j, i) + m(i, j)) * s;
			q[k] = (m(k, i) + m(i, k)) * s;
			v.x = q[0];
			v.y = q[1];
			v.z = q[2];
		}
	}

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

	QuaternionBase operator-() const {
		return QuaternionBase(-v, -w);
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

	MatrixND<3, 3, real, ISE> toMatrix3x3() const {
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
		return transpose(mat);		// use left handed
	}

	MatrixND<4, 4, real, ISE> toMatrix4x4() const {
		MatrixND<4, 4, real, ISE> mat;
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
		mat(3, 3) = 1_f;
		return transpose(mat);		// use left handed
	}

	friend std::ostream& operator<<(std::ostream& os, const QuaternionBase& q) {
		os << '[' << q.v.x << ',' << q.v.y << ',' << q.v.z << ',' << q.w << ']';
		return os;
	}

public:
	RealPart v;
	real w;
};

using Quaternion = QuaternionBase<default_instruction_set>;

PIMATH_NAMESPACE_END