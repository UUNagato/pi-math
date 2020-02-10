#include <iostream>
#include <chrono>

// #define PM_ISE_SSE

#include "include/pimath.h"
#include "include/advanced/pcgrng.h"

using namespace std;
using namespace PIMATH_NAMESPACE_NAME;

int main()
{
	//Matrix4 m1{ 2.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f };
	//Matrix4 m2{ 2.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f };
	Vector3 v1(1_f, 2_f, 0_f);

	Matrix3 m31{ 1_f, 0_f, 0_f, 0_f, 2_f, 0_f, 0_f, 0_f, 1_f };
	//Quaternion q(0.1, 0.3, 0.3, 1.0);
	//q = q.normalize();
	//v1.normalizeInPlace();

	std::cout << (m31 * v1);

	system("pause");
    return 0;
}