#include <iostream>
#include <chrono>

#define PM_ISE_SSE

#include "include/pimath.h"
#include "include/advanced/pcgrng.h"

using namespace std;
using namespace PIMATH_NAMESPACE_NAME;

int main()
{
	Matrix4 m1{ 2.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f };
	Vector3 v1(1_f, 2_f, 0_f);

	Vector3 v2(0_f, 1_f, 0_f);
	Quaternion q;
	v1.normalizeInPlace();

	std::cout << v1;

	system("pause");
    return 0;
}