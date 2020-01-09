#pragma once

// This is the header file for all basic data types

#include "basic/datatype.h"
#include "basic/array.h"
#include "basic/matrix.h"

NAMESPACE_PIMATH_BEGIN

typedef MatrixND<2, 1, float32> Vector2f;
typedef MatrixND<2, 1, float64> Vector2d;
typedef MatrixND<2, 1, real> Vector2;
typedef MatrixND<3, 1, float32> Vector3f;
typedef MatrixND<3, 1, float64> Vector3d;
typedef MatrixND<3, 1, real> Vector3;
typedef MatrixND<4, 1, float32> Vector4f;
typedef MatrixND<4, 1, float64> Vector4d;
typedef MatrixND<4, 1, real> Vector4;

typedef MatrixND<2, 2, float32> Matrix2f;
typedef MatrixND<2, 2, float64> Matrix2d;
typedef MatrixND<2, 2, real> Matrix2;
typedef MatrixND<3, 3, float32> Matrix3f;
typedef MatrixND<3, 3, float64> Matrix3d;
typedef MatrixND<3, 3, real> Matrix3;
typedef MatrixND<4, 4, float32> Matrix4f;
typedef MatrixND<4, 4, float64> Matrix4d;
typedef MatrixND<4, 4, real> Matrix4;

// more easy names
using Matrix2x2 = Matrix2;
using Matrix3x3 = Matrix3;
using Matrix4x4 = Matrix4;


NAMESPACE_PIMATH_END