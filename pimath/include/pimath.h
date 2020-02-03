#pragma once

// This is the header file for all basic data types
#include <limits>
#include "basic/datatype.h"
#include "basic/array.h"
#include "basic/matrix.h"
#include "basic/funcs.h"

NAMESPACE_PIMATH_BEGIN

using Vector2f = MatrixND<2, 1, float32>;
using Vector2d = MatrixND<2, 1, float64>;
using Vector2 = MatrixND<2, 1, real>;
template<typename T> using Vector2t = MatrixND<2, 1, T>;
using Vector3f = MatrixND<3, 1, float32>;
using Vector3d = MatrixND<3, 1, float64>;
using Vector3 = MatrixND<3, 1, real>;
template<typename T> using Vector3t = MatrixND<3, 1, T>;
using Vector4f = MatrixND<4, 1, float32>;
using Vector4d = MatrixND<4, 1, float64>;
using Vector4 = MatrixND<4, 1, real>;
template<typename T> using Vector4t = MatrixND<4, 1, T>;

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

// some special value
static constexpr real RInfinity = std::numeric_limits<real>::infinity();
static constexpr real RMax = std::numeric_limits<real>::max();
static constexpr real RLowest = std::numeric_limits<real>::lowest();


NAMESPACE_PIMATH_END