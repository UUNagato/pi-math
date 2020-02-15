/*
	Matrix Transform Toolset
	Define functions for quick matrix generation
*/

#pragma once

#include <cmath>
#include "datatype.h"
#include "matrix.h"

PIMATH_NAMESPACE_BEGIN

template<InstSetExt ISE = default_instruction_set>
MatrixND<4, 4, real, ISE> Translate(real x, real y, real z, MatrixND<4, 4, real, ISE> *inv = nullptr)
{
    MatrixND<4, 4, real, ISE> m{ 1_f, 0_f, 0_f, x,
                                0_f, 1_f, 0_f, y,
                                0_f, 0_f, 1_f, z,
                                0_f, 0_f, 0_f, 1_f };
    if (inv) {
        *inv = m;
        (*inv)[3] = -m[3];
        (*inv)[3][3] = 1_f;
    }
    return m;
}

template<InstSetExt ISE = default_instruction_set>
MatrixND<4, 4, real, ISE> Scale(real x, real y, real z, MatrixND<4, 4, real, ISE>* inv = nullptr)
{
    MatrixND<4, 4, real, ISE> m{ x, 0_f, 0_f, 0_f,
                                0_f, y, 0_f, 0_f,
                                0_f, 0_f, z, 0_f,
                                0_f, 0_f, 0_f, 1_f };
    if (inv) {
        *inv = m;
        (*inv)[0][0] = 1_f / x;
        (*inv)[1][1] = 1_f / y;
        (*inv)[2][2] = 1_f / z;
    }
    return m;
}

PIMATH_NAMESPACE_END