#pragma once

#include "StormSTL.h"

#include <cstddef>
#include <cstdint>

#include "StlDiagnostics.h"

#if STL_SSE_SUPPORT || STL_AVX_SUPPORT || STL_AVX512_SUPPORT
#if defined(_MSC_VER)
#include <immintrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif
#endif

namespace Stl::V {
	enum class STL_RUNTIME_API VType : uint8_t {
#if STL_SSE_SUPPORT
		V_SSE = 0,
#endif
#if STL_AVX_SUPPORT
		V_AVX = 1,
#endif
#if STL_AVX512_SUPPORT
		V_AVX512 = 2,
#endif
		V_UNKNOWN = 255
	};

	template<VType Type = VType::V_UNKNOWN, typename T = void>
	struct TResolve final {
		using resolve = std::conditional_t<std::is_void_v<T>, void, T>;
	};

	template<VType Type = VType::V_UNKNOWN, typename T = void, size_t v = 0>
	struct TMaskResolve final {
		using resolve = typename TResolve<Type, T>::resolve;
	};

	template<typename T>
	struct IsInteger final {
		static constexpr bool value = std::is_integral_v<T>;
	};

	template<typename T>
	struct IsFloat final {
		static constexpr bool value = std::is_same_v<T, float>;
	};

	template<typename T>
	struct IsDouble final {
		static constexpr bool value = std::is_same_v<T, double>;
	};

	template<bool B, typename T, typename V>
	using cond_v = std::conditional_t<B, T, V>;

	template<VType Type = VType::V_UNKNOWN>
	struct VIntrospect final {
		static constexpr size_t kWidth = 0;
		static constexpr size_t kAlign = 0;
		static constexpr auto kName = "Unknown";

		template<typename T>
		using RType = typename TResolve<Type, T>::resolve;

		template<typename T, size_t MaskWidth = 0>
		using MType = typename TMaskResolve<Type, T, MaskWidth>::resolve;
	};

#if STL_SSE_SUPPORT
	template<typename T>
	struct TResolve<VType::V_SSE, T> final {
		using resolve = cond_v<IsInteger<T>::value, __m128i, cond_v<IsFloat<T>::value, __m128, cond_v<IsDouble<T>::value, __m128d, void>>>;
	};

	template<>
	struct VIntrospect<VType::V_SSE> final {
		static constexpr size_t kWidth = 16;
		static constexpr size_t kAlign = 16;
		static constexpr auto kName = "SSE";

		template<typename T>
		using RType = typename TResolve<VType::V_SSE, T>::resolve;

		template<typename T, size_t MaskWidth = 0>
		using MType = typename TMaskResolve<VType::V_SSE, T, MaskWidth>::resolve;
	};
#endif

#if STL_AVX_SUPPORT
	template<typename T>
	struct TResolve<VType::V_AVX, T> final {
		using resolve = cond_v<IsInteger<T>::value, __m256i, cond_v<IsFloat<T>::value, __m256, cond_v<IsDouble<T>::value, __m256d, void>>>;
	};

	template<>
	struct VIntrospect<VType::V_AVX> final {
		static constexpr size_t kWidth = 32;
		static constexpr size_t kAlign = 32;
		static constexpr auto kName = "AVX";

		template<typename T>
		using RType = typename TResolve<VType::V_AVX, T>::resolve;

		template<typename T, size_t MaskWidth = 0>
		using MType = typename TMaskResolve<VType::V_AVX, T, MaskWidth>::resolve;
	};
#endif

#if STL_AVX512_SUPPORT
	template<typename T>
	struct TResolve<VType::V_AVX512, T> final {
		using resolve = cond_v<IsInteger<T>::value, __m512i, cond_v<IsFloat<T>::value, __m512, cond_v<IsDouble<T>::value, __m512d, void>>>;
	};

	template<typename T, size_t MaskWidth>
	struct TMaskResolve<VType::V_AVX512, T, MaskWidth> final {
		using resolve = cond_v<MaskWidth == 8, __mmask8, cond_v<MaskWidth == 16, __mmask16, cond_v<MaskWidth == 32, __mmask32, cond_v<MaskWidth == 64, __mmask64, void>>>>;
	};

	template<>
	struct VIntrospect<VType::V_AVX512> final {
		static constexpr size_t kWidth = 64;
		static constexpr size_t kAlign = 64;
		static constexpr auto kName = "AVX-512";

		template<typename T>
		using RType = typename TResolve<VType::V_AVX512, T>::resolve;

		template<typename T>
		using MType = typename TMaskResolve<VType::V_AVX512, T, kWidth / sizeof(T)>::resolve;
	};
#endif
}
