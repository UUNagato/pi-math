# PI-Math

A C++ header-only math library designed for Graphics.
**It is still under construction. And not recommend to use it now.**

## How to use
Please include the file pimath.h into your project and use namespace Pimath in order to use it.
You could use the following statement to use the namespace
```cpp
using namespace PIMATH_NAMESPACE_NAME;
```

## Datatype:
There are mainly three datatypes now in PI-Math: **Vector**, **Matrix** and **Quaternion**.
**Vector** is a sqecial case of **Matrix** (with only 1 row or colume). The data is stored colume-majored.

There is also a numeric type called **real**, which is a 32 bit float or 64 bit double number depending on whether or not the macro **PM_64BIT** is defined. There is also a literal value operator for **real**, for example:
```cpp
real a = 3.14_f
```

**Vector**, **Matrix** are the same type: **MatrixND<rows, cols, type, instruction set>**
You could use the predefined names for common data types.
**Vector\[ndim\]\[type\]**

For example, a 3 dimensional real vector is:
**Vector3**
.A 2 dimensional float vector is:
**Vector2f**.A 4 dimensional template type vector is:
**Vector4t\<T\>**

4x4 Matrix could be:
**Matrix4x4** or **Matrix4**

## Usage:
### Vector:
A Vector can be initialized with following ways
```cpp
Vector3 a; // all value 0
Vector3 b(1_f); // b = (1, 1, 1)
Vector3 c{ 1_f, 2_f, 3_f};  // c = (1,2,3)
Vector3 d(1_f, 2_f, 3_f);   // d = (1,2,3), this only works for vectors no longer than 4.
```

To access a member:
```cpp
a[i];   // access the element with index i (starting from 0)
a.x, a.y, a.z, a.w;   // used for vectors no longer than 4.
```
