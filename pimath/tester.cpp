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
	Quaternion q(0.1, 0.3, 0.3, 1.0);
	q = q.normalize();
	v1.normalizeInPlace();

	std::cout << q << std::endl;
	Matrix4x4 m = q.toMatrix4x4();
	std::cout << "Q Matrix:" << m << '\n';
	std::cout << "Matrix to Quaternion:" << Quaternion(m) << '\n';

	system("pause");
    return 0;
}