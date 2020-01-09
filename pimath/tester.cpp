#include <iostream>
#include <chrono>

#define PM_ISE_SSE

#include "include/pimath.h"

using namespace std;
using namespace NAMESPACE_PIMATH;

int main()
{
	Matrix4 m{ 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f };
	std::cout << inversed(m);
	system("pause");
    return 0;
}