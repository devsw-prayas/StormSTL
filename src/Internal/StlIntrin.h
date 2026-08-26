#pragma once
#include <immintrin.h>
#include "StlVUtils.h"

#ifndef STL_INVOCABLE
// clang-format off
#define STL_INVOCABLE(invocable)                                                                                                                                                                       \
	invocable(const invocable&) = delete;                                                                                                                                                              \
	invocable(invocable&&) noexcept = delete;                                                                                                                                                          \
	invocable& operator=(const invocable&) = delete;                                                                                                                                                   \
	invocable& operator=(invocable&&) noexcept = delete;                                                                                                                                               \
private:                                                                                                                                                                                               \
	invocable() = delete;                                                                                                                                                                              \
	~invocable() = delete;

// clang-format on
#endif

namespace Stl::Internal::V {
	using namespace Stl::V;

	// Canonical vectorized aligned-load
	template<VType Type = VType::V_UNKNOWN, typename T = void>
	struct Load final {
		STL_INVOCABLE(Load)
		using r = VIntrospect<Type>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(const T* p_Src) noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE_SUPPORT
	template<typename T>
	struct Load<VType::V_SSE, T> final {
		STL_INVOCABLE(Load)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(const T* p_Src) noexcept {
			if constexpr (IsInteger<T>::value) return _mm_load_si128(reinterpret_cast<const r*>(p_Src));
			if constexpr (IsFloat<T>::value) return _mm_load_ps(p_Src);
			if constexpr (IsDouble<T>::value) return _mm_load_pd(p_Src);
			STL_UNREACHABLE();
		}
	};

#endif

#if STL_AVX_SUPPORT
	template<typename T>
	struct Load<VType::V_AVX, T> final {
		STL_INVOCABLE(Load)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(const T* p_Src) noexcept {
			if constexpr (IsInteger<T>::value) return _mm256_load_si256(reinterpret_cast<const r*>(p_Src));
			if constexpr (IsFloat<T>::value) return _mm256_load_ps(p_Src);
			if constexpr (IsDouble<T>::value) return _mm256_load_pd(p_Src);
			STL_UNREACHABLE();
		}
	};

#endif

#if STL_AVX512_SUPPORT
	template<typename T>
	struct Load<VType::V_AVX512, T> final {
		STL_INVOCABLE(Load)
		using r = VIntrospect<VType::V_AVX512>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(const T* p_Src) noexcept {
			if constexpr (IsInteger<T>::value) return _mm512_load_si512(reinterpret_cast<const r*>(p_Src));
			if constexpr (IsFloat<T>::value) return _mm512_load_ps(p_Src);
			if constexpr (IsDouble<T>::value) return _mm512_load_pd(p_Src);
			STL_UNREACHABLE();
		}
	};

#endif

	// Canonical vectorized unaligned-load
	template<VType Type = VType::V_UNKNOWN, typename T = void>
	struct Loadu final {
		STL_INVOCABLE(Loadu)
		using r = VIntrospect<Type>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(const T* p_Src) noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE_SUPPORT
	template<typename T>
	struct Loadu<VType::V_SSE, T> final {
		STL_INVOCABLE(Loadu)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(const T* p_Src) noexcept {
			if constexpr (IsInteger<T>::value) return _mm_loadu_si128(reinterpret_cast<const r*>(p_Src));
			if constexpr (IsFloat<T>::value) return _mm_loadu_ps(p_Src);
			if constexpr (IsDouble<T>::value) return _mm_loadu_pd(p_Src);
			STL_UNREACHABLE();
		}
	};

#endif

#if STL_AVX_SUPPORT
	template<typename T>
	struct Loadu<VType::V_AVX, T> final {
		STL_INVOCABLE(Loadu)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(const T* p_Src) noexcept {
			if constexpr (IsInteger<T>::value) return _mm256_loadu_si256(reinterpret_cast<const r*>(p_Src));
			if constexpr (IsFloat<T>::value) return _mm256_loadu_ps(p_Src);
			if constexpr (IsDouble<T>::value) return _mm256_loadu_pd(p_Src);
			STL_UNREACHABLE();
		}
	};

#endif

#if STL_AVX512_SUPPORT
	template<typename T>
	struct Loadu<VType::V_AVX512, T> final {
		STL_INVOCABLE(Loadu)
		using r = VIntrospect<VType::V_AVX512>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(const T* p_Src) noexcept {
			if constexpr (IsInteger<T>::value) return _mm512_loadu_si512(reinterpret_cast<const r*>(p_Src));
			if constexpr (IsFloat<T>::value) return _mm512_loadu_ps(p_Src);
			if constexpr (IsDouble<T>::value) return _mm512_loadu_pd(p_Src);
			STL_UNREACHABLE();
		}
	};

#endif

	template<VType Type, typename T>
	struct Store final {
		STL_INVOCABLE(Store)
		using r = VIntrospect<Type>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE_SUPPORT
	template<typename T>
	struct Store<VType::V_SSE, T> final {
		STL_INVOCABLE(Store)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm_store_si128(reinterpret_cast<__m128i*>(p_Dst), v_Reg);
			else if constexpr (IsFloat<T>::value) _mm_store_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm_store_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};

#endif

#if STL_AVX_SUPPORT
	template<typename T>
	struct Store<VType::V_AVX, T> final {
		STL_INVOCABLE(Store)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm256_store_si256(reinterpret_cast<__m256i*>(p_Dst), v_Reg);
			else if constexpr (IsFloat<T>::value) _mm256_store_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm256_store_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};

#endif

#if STL_AVX512_SUPPORT
	template<typename T>
	struct Store<VType::V_AVX512, T> final {
		STL_INVOCABLE(Store)
		using r = VIntrospect<VType::V_AVX512>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm512_store_si512(p_Dst, v_Reg);
			else if constexpr (IsFloat<T>::value) _mm512_store_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm512_store_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};

#endif

	template<VType Type, typename T>
	struct Storeu final {
		STL_INVOCABLE(Storeu)
		using r = VIntrospect<Type>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE_SUPPORT
	template<typename T>
	struct Storeu<VType::V_SSE, T> final {
		STL_INVOCABLE(Storeu)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm_storeu_si128(reinterpret_cast<__m128i*>(p_Dst), v_Reg);
			else if constexpr (IsFloat<T>::value) _mm_storeu_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm_storeu_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};

#endif

#if STL_AVX_SUPPORT
	template<typename T>
	struct Storeu<VType::V_AVX, T> final {
		STL_INVOCABLE(Storeu)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_Dst), v_Reg);
			else if constexpr (IsFloat<T>::value) _mm256_storeu_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm256_storeu_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};

#endif

#if STL_AVX512_SUPPORT
	template<typename T>
	struct Storeu<VType::V_AVX512, T> final {
		STL_INVOCABLE(Storeu)
		using r = VIntrospect<VType::V_AVX512>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm512_storeu_si512(p_Dst, v_Reg);
			else if constexpr (IsFloat<T>::value) _mm512_storeu_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm512_storeu_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};

#endif

	// Canonical vectorized aligned non-temporal-store
	template<VType Type, typename T>
	struct Stream final {
		STL_INVOCABLE(Stream)
		using r = VIntrospect<Type>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE_SUPPORT
	template<typename T>
	struct Stream<VType::V_SSE, T> final {
		STL_INVOCABLE(Stream)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm_stream_si128(reinterpret_cast<__m128i*>(p_Dst), v_Reg);
			else if constexpr (IsFloat<T>::value) _mm_stream_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm_stream_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX_SUPPORT
	template<typename T>
	struct Stream<VType::V_AVX, T> final {
		STL_INVOCABLE(Stream)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm256_stream_si256(reinterpret_cast<__m256i*>(p_Dst), v_Reg);
			else if constexpr (IsFloat<T>::value) _mm256_stream_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm256_stream_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX512_SUPPORT
	template<typename T>
	struct Stream<VType::V_AVX512, T> final {
		STL_INVOCABLE(Stream)
		using r = VIntrospect<VType::V_AVX512>::template RType<T>;

	public:
		static STL_FORCEINLINE void invoke(T* p_Dst, r v_Reg) noexcept {
			if constexpr (IsInteger<T>::value) _mm512_stream_si512(p_Dst, v_Reg);
			else if constexpr (IsFloat<T>::value) _mm512_stream_ps(p_Dst, v_Reg);
			else if constexpr (IsDouble<T>::value) _mm512_stream_pd(p_Dst, v_Reg);
			else STL_UNREACHABLE();
		}
	};
#endif

	// Store-ordering fence for the x86 TSO memory model.
	template<VType Type = VType::V_UNKNOWN>
	struct Fence final {
		STL_INVOCABLE(Fence)

	public:
		static STL_FORCEINLINE void invoke() noexcept {
			_mm_sfence();
		}
	};
}