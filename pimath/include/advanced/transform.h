#pragma once
#include "../basic/utils.h"
#include "../pimath.h"

NAMESPACE_PIMATH_BEGIN

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
		Vector4 vec{ normal.x, normal.y, normal.z, 1_f };
		Vector4 transformed = inv_m.transpose() * vec;
		return Vector3(transformed).normalize();
	}

private:
	Matrix4x4 m;
	Matrix4x4 inv_m;
};

NAMESPACE_PIMATH_END