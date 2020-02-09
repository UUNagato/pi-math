#pragma once
#include "../basic/utils.h"
#include "../pimath.h"

PIMATH_NAMESPACE_BEGIN

// Transform Class, a tool class for transformations.

class Transform
{
public:
	Transform() {
	// make m identity
		for (int i = 0; i < 4; ++i) {
			m(i, i) = 1_f;
			inv_m(i, i) = 1_f;
		}
	}

	~Transform() {}

	Vector3 applyNormal(const Vector3& normal) {
		Vector3 transformed = inv_m.transpose().applyTransform(normal);
		return Vector3(transformed).normalize();
	}

	Vector3 transformPoint(const Vector3& point) {
		Vector4 p(point, 1_f);
		Vector4 transformed = m * p;
		return Vector3(p);
	}

	Vector3 transformDirection(const Vector3& dir) {
		Vector3 transformed = m.applyTransform(dir);
		return transformed;
	}

	void translate(const Vector3& distance) {
		m(0, 3) += distance.x;
		m(1, 3) += distance.y;
		m(2, 3) += distance.z;
		inv_m(0, 3) += -distance.x;
		inv_m(1, 3) += -distance.y;
		inv_m(2, 3) += -distance.z;
	}

private:
	Matrix4x4 m;
	Matrix4x4 inv_m;
};

PIMATH_NAMESPACE_END