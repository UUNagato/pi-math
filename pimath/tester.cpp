#include <iostream>

// #define PM_ISE_SSE
#include "include/basic/array.h"
#include "include/basic/matrix.h"

using namespace std;

int main()
{
    Pimath::MatrixND<3, 3, float> m1{ 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f };
    Pimath::MatrixND<2, 2, float> m2{ 1.f, 0.f, 1.f, 0.f };
	m1 += 1.f;
    std::cout << m1;

	system("pause");
    return 0;
}