#include <iostream>

// #define PM_ISE_SSE
#include "include/basic/vector.h"
#include "include/basic/matrix.h"

using namespace std;

int main()
{
    Pimath::MatrixND<3, 3, float> m1{ 1.2f, 1.1f, 2.f, 1.f, 1.3f, 15.f, 2.f, 2.3f, 2.4f };
    Pimath::MatrixND<3, 3, float> m2{ 2.2f, 3.1f, 2.3f, 1.2f, 1.5f, 15.f, 2.f, 2.3f, 2.4f };
    std::cout << m1 + m2;
    return 0;
}