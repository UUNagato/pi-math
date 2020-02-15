#include <iostream>
#include <chrono>

#define PM_ISE_SSE

#include "include/pimath.h"
#include "include/basic/matrix_transform.h"
#include "include/advanced/mat_special.h"
#include "include/advanced/pcgrng.h"
#include "include/basic/funcs.h"

struct TEST_S
{
	union {
		__m128 v;
		struct {
			float x, y, z, w;
		};
	};
};

using namespace std;
using namespace PIMATH_NAMESPACE_NAME;

int main()
{
	//Matrix4 m1{ 2.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f };
	//Matrix4 m2{ 2.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f };
	Matrix4 s = Scale(2_f, 2_f, 2_f);
	Matrix4 s2 = Scale(1_f, 2_f, 2_f);
	Vector3 n(1_f, 1_f, 1_f);
	Vector3 n2(1_f, 2_f, 1_f);
	Quaternion q, p;
	std::cout << ceil(n + n2);
	system("pause");
    return 0;
}