#include <iostream>
#include <chrono>

#define PM_ISE_SSE

#include "include/pimath.h"

using namespace std;
using namespace NAMESPACE_PIMATH;

int main()
{
	Matrix4 m1{ 2.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f };
	Vector3 v1{ 1.f, 2.f, 0.5f };

	std::cout << m1.applyTransform(v1);
	system("pause");
    return 0;
}