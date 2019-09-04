#pragma once
#define NAMESPACE_PIMATH Pimath

#define NAMESPACE_PIMATH_BEGIN namespace Pimath {
#define NAMESPACE_PIMATH_END }

// used for compile bits
#define PIMATH_64BIT

NAMESPACE_PIMATH_BEGIN
// Instruction Set Extension, learnt from Taichi (https://github.com/yuanming-hu/taichi)
enum class InstSetExt { None, SSE, AVX, AVX2 };

#ifdef PM_ISE_NONE
constexpr InstSetExt default_instruction_set = InstSetExt::None;
#elif defined(PM_ISE_SSE)
constexpr InstSetExt default_instruction_set = InstSetExt::SSE;
#elif defined(PM_ISE_AVX)
constexpr InstSetExt default_instruction_set = InstSetExt::AVX;
#elif defined(PM_ISE_AVX2)
constexpr InstSetExt default_instruction_set = InstSetExt::AVX2;
#else
#define PM_ISE_NONE
constexpr InstSetExt default_instruction_set = InstSetExt::None;
#endif

// Inline define
#define PM_INLINE inline

// Data Aligned
#if defined(_MSC_VER)
#define PM_ALIGNED(x) __declspec(align(x))
#else
#define PM_ALIGNED(x) __attribute__((aligned(x)))
#endif

NAMESPACE_PIMATH_END