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

	// Canonical vectorized zero-register construction.
	template<VType Type, typename T>
	struct SetZero final {
		STL_INVOCABLE(SetZero)
		using r = VIntrospect<Type>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke() noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE_SUPPORT
	template<typename T>
	struct SetZero<VType::V_SSE, T> final {
		STL_INVOCABLE(SetZero)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke() noexcept {
			if constexpr (IsInteger<T>::value) return _mm_setzero_si128();
			else if constexpr (IsFloat<T>::value) return _mm_setzero_ps();
			else if constexpr (IsDouble<T>::value) return _mm_setzero_pd();
			else STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX_SUPPORT
	template<typename T>
	struct SetZero<VType::V_AVX, T> final {
		STL_INVOCABLE(SetZero)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke() noexcept {
			if constexpr (IsInteger<T>::value) return _mm256_setzero_si256();
			else if constexpr (IsFloat<T>::value) return _mm256_setzero_ps();
			else if constexpr (IsDouble<T>::value) return _mm256_setzero_pd();
			else STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX512_SUPPORT
	template<typename T>
	struct SetZero<VType::V_AVX512, T> final {
		STL_INVOCABLE(SetZero)
		using r = VIntrospect<VType::V_AVX512>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke() noexcept {
			if constexpr (IsInteger<T>::value) return _mm512_setzero_si512();
			else if constexpr (IsFloat<T>::value) return _mm512_setzero_ps();
			else if constexpr (IsDouble<T>::value) return _mm512_setzero_pd();
			else STL_UNREACHABLE();
		}
	};
#endif

	// Canonical vectorized scalar broadcast.
	template<VType Type, typename T>
	struct Set1 final {
		STL_INVOCABLE(Set1)
		using r = VIntrospect<Type>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(T v_Value) noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE_SUPPORT
	template<typename T>
	struct Set1<VType::V_SSE, T> final {
		STL_INVOCABLE(Set1)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(T v_Value) noexcept {
			if constexpr (IsInteger<T>::value) {
				if constexpr (sizeof(T) == 1) return _mm_set1_epi8(static_cast<char>(v_Value));
				else if constexpr (sizeof(T) == 2) return _mm_set1_epi16(static_cast<short>(v_Value));
				else if constexpr (sizeof(T) == 4) return _mm_set1_epi32(static_cast<int>(v_Value));
				else if constexpr (sizeof(T) == 8) return _mm_set1_epi64x(static_cast<long long>(v_Value));
			}
			else if constexpr (IsFloat<T>::value) return _mm_set1_ps(v_Value);
			else if constexpr (IsDouble<T>::value) return _mm_set1_pd(v_Value);
			STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX_SUPPORT
	template<typename T>
	struct Set1<VType::V_AVX, T> final {
		STL_INVOCABLE(Set1)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(T v_Value) noexcept {
			if constexpr (IsInteger<T>::value) {
				if constexpr (sizeof(T) == 1) return _mm256_set1_epi8(static_cast<char>(v_Value));
				else if constexpr (sizeof(T) == 2) return _mm256_set1_epi16(static_cast<short>(v_Value));
				else if constexpr (sizeof(T) == 4) return _mm256_set1_epi32(static_cast<int>(v_Value));
				else if constexpr (sizeof(T) == 8) return _mm256_set1_epi64x(static_cast<long long>(v_Value));
			}
			else if constexpr (IsFloat<T>::value) return _mm256_set1_ps(v_Value);
			else if constexpr (IsDouble<T>::value) return _mm256_set1_pd(v_Value);
			STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX512_SUPPORT
	template<typename T>
	struct Set1<VType::V_AVX512, T> final {
		STL_INVOCABLE(Set1)
		using r = VIntrospect<VType::V_AVX512>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(T v_Value) noexcept {
			if constexpr (IsInteger<T>::value) {
				if constexpr (sizeof(T) == 1) return _mm512_set1_epi8(static_cast<char>(v_Value));
				else if constexpr (sizeof(T) == 2) return _mm512_set1_epi16(static_cast<short>(v_Value));
				else if constexpr (sizeof(T) == 4) return _mm512_set1_epi32(static_cast<int>(v_Value));
				else if constexpr (sizeof(T) == 8) return _mm512_set1_epi64(static_cast<long long>(v_Value));
			}
			else if constexpr (IsFloat<T>::value) return _mm512_set1_ps(v_Value);
			else if constexpr (IsDouble<T>::value) return _mm512_set1_pd(v_Value);
			STL_UNREACHABLE();
		}
	};
#endif

	// Canonical byte-wise vector equality comparison.
	template<VType Type, typename T>
	struct CompareEqual final {
		STL_INVOCABLE(CompareEqual)
		using r = VIntrospect<Type>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(r v_Left, r v_Right) noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE2_SUPPORT
	template<typename T>
	struct CompareEqual<VType::V_SSE, T> final {
		STL_INVOCABLE(CompareEqual)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(r v_Left, r v_Right) noexcept {
			if constexpr (sizeof(T) == 1 && IsInteger<T>::value) return _mm_cmpeq_epi8(v_Left, v_Right);
			else STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX2_SUPPORT
	template<typename T>
	struct CompareEqual<VType::V_AVX, T> final {
		STL_INVOCABLE(CompareEqual)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE r invoke(r v_Left, r v_Right) noexcept {
			if constexpr (sizeof(T) == 1 && IsInteger<T>::value) return _mm256_cmpeq_epi8(v_Left, v_Right);
			else STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX512BW_SUPPORT
	template<typename T>
	struct CompareEqual<VType::V_AVX512, T> final {
		STL_INVOCABLE(CompareEqual)
		using m = VIntrospect<VType::V_AVX512>::template MType<T>;
		using r = VIntrospect<VType::V_AVX512>::template RType<T>;

	public:
		static STL_FORCEINLINE m invoke(r v_Left, r v_Right) noexcept {
			if constexpr (sizeof(T) == 1 && IsInteger<T>::value) return _mm512_cmpeq_epi8_mask(v_Left, v_Right);
			else STL_UNREACHABLE();
		}
	};
#endif

	// Canonical extraction of byte sign bits into a scalar/backend mask.
	template<VType Type, typename T>
	struct MoveMask final {
		STL_INVOCABLE(MoveMask)

	public:
		static STL_FORCEINLINE uint64_t invoke(VIntrospect<Type>::template RType<T> v_Reg) noexcept {
			STL_UNREACHABLE();
		}
	};

#if STL_SSE2_SUPPORT
	template<typename T>
	struct MoveMask<VType::V_SSE, T> final {
		STL_INVOCABLE(MoveMask)
		using r = VIntrospect<VType::V_SSE>::template RType<T>;

	public:
		static STL_FORCEINLINE uint64_t invoke(r v_Reg) noexcept {
			if constexpr (sizeof(T) == 1 && IsInteger<T>::value) return static_cast<uint32_t>(_mm_movemask_epi8(v_Reg));
			else STL_UNREACHABLE();
		}
	};
#endif

#if STL_AVX2_SUPPORT
	template<typename T>
	struct MoveMask<VType::V_AVX, T> final {
		STL_INVOCABLE(MoveMask)
		using r = VIntrospect<VType::V_AVX>::template RType<T>;

	public:
		static STL_FORCEINLINE uint64_t invoke(r v_Reg) noexcept {
			if constexpr (sizeof(T) == 1 && IsInteger<T>::value) return static_cast<uint32_t>(_mm256_movemask_epi8(v_Reg));
			else STL_UNREACHABLE();
		}
	};
#endif

	// Canonical software prefetch operations.
	struct PrefetchRead final {
		STL_INVOCABLE(PrefetchRead)

	public:
		template<int Locality>
		static STL_FORCEINLINE void invoke(const void* p_Ptr) noexcept {
			static_assert(Locality >= 0 && Locality <= 3, "Invalid prefetch locality");
			_mm_prefetch(reinterpret_cast<const char*>(p_Ptr), Locality);
		}
	};

	struct PrefetchWrite final {
		STL_INVOCABLE(PrefetchWrite)

	public:
		static STL_FORCEINLINE void invoke(const void* p_Ptr) noexcept {
#if STL_COMPILER_MSVC
			_m_prefetchw(const_cast<void*>(p_Ptr));
#else
			__builtin_prefetch(p_Ptr, 1, 3);
#endif
		}
	};

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
