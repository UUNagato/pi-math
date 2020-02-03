#include <iostream>
#include <chrono>

#define PM_ISE_SSE

#include "include/pimath.h"

using namespace std;
using namespace NAMESPACE_PIMATH;

int main()
{
	Matrix4 m1{ 2.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f };
	Vector3 v1(1_f, 2_f, 3_f);

	Vector3 v2(0.5_f, 2_f, 2_f);

	std::cout << (v1 * v2);
	system("pause");
    return 0;
}