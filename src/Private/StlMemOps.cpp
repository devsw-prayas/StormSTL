#include "StormSTL.h"
#include "StlMemOps.h"
#include "StlCpu.h"

#define ALLOW_SYSCALL

#include "StlSyscalls.h"

namespace Stl::Memory {
	using Stl::Internal::nonTemporalThreshold;
	using Stl::Internal::stosThreshold;

	void memSet(void* STL_RESTRICT p_Dst, uint8_t v_Val, size_t v_Size) {
		using introspect = V::VIntrospect<V::preferredBackend()>;
		using r = introspect::RType<uint8_t>;
		constexpr size_t W = introspect::kWidth;
		constexpr uintptr_t mask = introspect::kAlign - 1;

		if (v_Size == 0) return;

		if (v_Size < W || v_Size <= nonTemporalThreshold()) {
#if STL_COMPILER_MSVC
			__stosb(static_cast<uint8_t*>(p_Dst), v_Val, v_Size);
#else
			__builtin_memset(p_Dst, v_Val, v_Size);
#endif
			return;
		}

		auto* const d = static_cast<uint8_t*>(p_Dst);
		const uintptr_t p = reinterpret_cast<uintptr_t>(d);
		const r l = Internal::Set1<V::preferredBackend(), uint8_t>::invoke(v_Val);

		Internal::Storeu<V::preferredBackend(), uint8_t>::invoke(d, l);

		auto* const s = std::bit_cast<uint8_t*>((p + mask) & ~mask);
		auto* const e = std::bit_cast<uint8_t*>((p + v_Size) & ~mask);

		Internal::streamWindowLoop<V::preferredBackend()>(s, static_cast<size_t>(e - s), l);
		Internal::Fence<V::preferredBackend()>::invoke();

		Internal::Storeu<V::preferredBackend(), uint8_t>::invoke(d + v_Size - W, l);
	}

	void memCopy(void* STL_RESTRICT p_Dst, const void* STL_RESTRICT p_Src, size_t v_Size) {
		constexpr auto backend = V::preferredBackend();
		using introspect = V::VIntrospect<backend>;
		constexpr size_t advance = introspect::kWidth * introspect::UType::value;
		constexpr uintptr_t align = introspect::kAlign - 1;

		if (v_Size == 0) return;

		auto* const dst = static_cast<uint8_t*>(p_Dst);
		const auto* const src = static_cast<const uint8_t*>(p_Src);

		// A copy touches src + dst, so its cache footprint is twice the payload.
		if (v_Size < advance || 2 * v_Size <= nonTemporalThreshold()) {
#if STL_COMPILER_MSVC
			__movsb(dst, src, v_Size);
#else
			__builtin_memcpy(dst, src, v_Size);
#endif
			return;
		}

		const uintptr_t d = reinterpret_cast<uintptr_t>(dst);
		const uintptr_t s = reinterpret_cast<uintptr_t>(src);

		size_t done;
		if (((d | s) & align) == 0) {
			done = v_Size - (v_Size % advance);
			Internal::memCopyStreamDispatch<backend>(src, dst, done);
		}
		else {
			// NT stores need an aligned dst; src offset may differ, so it rides along under Loadu.
			Internal::memCopyUnalignedDispatch<backend>(src, dst, advance);

			const size_t skip = (align + 1 - (d & align)) & align;
			const size_t rem = v_Size - skip;
			done = skip + rem - (rem % advance);
			Internal::memCopyStreamSrcUnalignedDispatch<backend>(src + skip, dst + skip, done - skip);
		}
		Internal::Fence<backend>::invoke();

		if (done != v_Size) {
			const size_t off = v_Size - advance;
			Internal::memCopyUnalignedDispatch<backend>(src + off, dst + off, advance);
		}
	}

	// TODO: implement.
	void memMove(void*, const void*, size_t) {

	}

	namespace {
		// Inputs shorter than two backend vectors: two overlapping loads (head + tail) of the
		// largest width that fits, so every size costs a fixed handful of instructions.

		template<typename T>
		STL_FORCEINLINE T loadScalar(const uint8_t* p_Src) {
			T v;
			std::memcpy(&v, p_Src, sizeof(T));
			return v;
		}

		template<typename T>
		STL_FORCEINLINE T byteSwap(T v_X) {
#if STL_COMPILER_MSVC
			if constexpr (sizeof(T) == 8) return _byteswap_uint64(v_X);
			else return _byteswap_ulong(v_X);
#else
			if constexpr (sizeof(T) == 8) return __builtin_bswap64(v_X);
			else return __builtin_bswap32(v_X);
#endif
		}

		// Loads are little-endian, so byte-swapping makes integer order match memory order.
		template<typename T>
		STL_FORCEINLINE int compareScalarPair(const uint8_t* p_A, const uint8_t* p_B, size_t v_Size) {
			T x = loadScalar<T>(p_A), y = loadScalar<T>(p_B);
			if (x == y) {
				x = loadScalar<T>(p_A + v_Size - sizeof(T));
				y = loadScalar<T>(p_B + v_Size - sizeof(T));
				if (x == y) return 0;
			}
			return byteSwap(x) < byteSwap(y) ? -1 : 1;
		}

		// SWAR zero-byte scan: the lowest flagged byte is exact (borrows only propagate upward).
		template<typename T>
		STL_FORCEINLINE size_t findScalar(T v_X, uint8_t v_Byte) {
			constexpr T ones = static_cast<T>(~T(0) / 0xFF);
			const T z = v_X ^ (ones * v_Byte);
			const T t = (z - ones) & ~z & (ones << 7);
			return t ? static_cast<size_t>(std::countr_zero(t)) / 8 : SIZE_MAX;
		}

		template<V::VType Type>
		STL_FORCEINLINE uint64_t vecMissBits(const uint8_t* p_A, const uint8_t* p_B) {
			using namespace Internal;
			constexpr size_t w = V::VIntrospect<Type>::kWidth;
			constexpr uint64_t full = w == 64 ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
			return ~byteEqualMask<Type>(Loadu<Type, uint8_t>::invoke(p_A), Loadu<Type, uint8_t>::invoke(p_B)) & full;
		}

		template<V::VType Type>
		STL_FORCEINLINE bool vecPairEqual(const uint8_t* p_A, const uint8_t* p_B, size_t v_Size) {
			using namespace Internal;
			constexpr size_t w = V::VIntrospect<Type>::kWidth;
			const auto lo = BitXor<Type>::invoke(Loadu<Type, uint8_t>::invoke(p_A), Loadu<Type, uint8_t>::invoke(p_B));
			const auto hi = BitXor<Type>::invoke(Loadu<Type, uint8_t>::invoke(p_A + v_Size - w), Loadu<Type, uint8_t>::invoke(p_B + v_Size - w));
			return TestZero<Type>::invoke(BitOr<Type>::invoke(lo, hi));
		}

		template<V::VType Type>
		STL_FORCEINLINE int vecPairCompare(const uint8_t* p_A, const uint8_t* p_B, size_t v_Size) {
			constexpr size_t w = V::VIntrospect<Type>::kWidth;
			size_t at;
			if (const uint64_t m = vecMissBits<Type>(p_A, p_B)) at = std::countr_zero(m);
			else if (const uint64_t m2 = vecMissBits<Type>(p_A + v_Size - w, p_B + v_Size - w)) at = v_Size - w + std::countr_zero(m2);
			else return 0;
			return p_A[at] < p_B[at] ? -1 : 1;
		}

		template<V::VType Type>
		STL_FORCEINLINE const uint8_t* vecPairFind(const uint8_t* p_Ptr, size_t v_Size, uint8_t v_Byte) {
			using namespace Internal;
			constexpr size_t w = V::VIntrospect<Type>::kWidth;
			const auto needle = Set1<Type, uint8_t>::invoke(v_Byte);
			if (const uint64_t m = byteEqualMask<Type>(Loadu<Type, uint8_t>::invoke(p_Ptr), needle)) return p_Ptr + std::countr_zero(m);
			if (const uint64_t m = byteEqualMask<Type>(Loadu<Type, uint8_t>::invoke(p_Ptr + v_Size - w), needle)) return p_Ptr + v_Size - w + std::countr_zero(m);
			return nullptr;
		}

		bool smallEqual(const uint8_t* p_A, const uint8_t* p_B, size_t v_Size) {
#if STL_AVX512BW_SUPPORT
			if (v_Size >= 64) return vecPairEqual<V::VType::V_AVX512>(p_A, p_B, v_Size);
#endif
#if STL_AVX2_SUPPORT
			if (v_Size >= 32) return vecPairEqual<V::VType::V_AVX>(p_A, p_B, v_Size);
#endif
			if (v_Size >= 16) return vecPairEqual<V::VType::V_SSE>(p_A, p_B, v_Size);
			if (v_Size >= 8)
				return ((loadScalar<uint64_t>(p_A) ^ loadScalar<uint64_t>(p_B)) |
					(loadScalar<uint64_t>(p_A + v_Size - 8) ^ loadScalar<uint64_t>(p_B + v_Size - 8))) == 0;
			if (v_Size >= 4)
				return ((loadScalar<uint32_t>(p_A) ^ loadScalar<uint32_t>(p_B)) |
					(loadScalar<uint32_t>(p_A + v_Size - 4) ^ loadScalar<uint32_t>(p_B + v_Size - 4))) == 0;
			if (v_Size >= 2)
				return ((loadScalar<uint16_t>(p_A) ^ loadScalar<uint16_t>(p_B)) |
					(loadScalar<uint16_t>(p_A + v_Size - 2) ^ loadScalar<uint16_t>(p_B + v_Size - 2))) == 0;
			return v_Size == 0 || *p_A == *p_B;
		}

		int smallCompare(const uint8_t* p_A, const uint8_t* p_B, size_t v_Size) {
#if STL_AVX512BW_SUPPORT
			if (v_Size >= 64) return vecPairCompare<V::VType::V_AVX512>(p_A, p_B, v_Size);
#endif
#if STL_AVX2_SUPPORT
			if (v_Size >= 32) return vecPairCompare<V::VType::V_AVX>(p_A, p_B, v_Size);
#endif
			if (v_Size >= 16) return vecPairCompare<V::VType::V_SSE>(p_A, p_B, v_Size);
			if (v_Size >= 8) return compareScalarPair<uint64_t>(p_A, p_B, v_Size);
			if (v_Size >= 4) return compareScalarPair<uint32_t>(p_A, p_B, v_Size);
			if (v_Size == 0) return 0;
			// 1..3 bytes: [0], [n/2], [n-1] covers every byte in order, packed big-endian.
			const uint32_t x = (uint32_t(p_A[0]) << 16) | (uint32_t(p_A[v_Size >> 1]) << 8) | p_A[v_Size - 1];
			const uint32_t y = (uint32_t(p_B[0]) << 16) | (uint32_t(p_B[v_Size >> 1]) << 8) | p_B[v_Size - 1];
			return (x > y) - (x < y);
		}

		const uint8_t* smallFind(const uint8_t* p_Ptr, size_t v_Size, uint8_t v_Byte) {
#if STL_AVX512BW_SUPPORT
			if (v_Size >= 64) return vecPairFind<V::VType::V_AVX512>(p_Ptr, v_Size, v_Byte);
#endif
#if STL_AVX2_SUPPORT
			if (v_Size >= 32) return vecPairFind<V::VType::V_AVX>(p_Ptr, v_Size, v_Byte);
#endif
			if (v_Size >= 16) return vecPairFind<V::VType::V_SSE>(p_Ptr, v_Size, v_Byte);
			if (v_Size >= 8) {
				if (const size_t i = findScalar(loadScalar<uint64_t>(p_Ptr), v_Byte); i != SIZE_MAX) return p_Ptr + i;
				if (const size_t i = findScalar(loadScalar<uint64_t>(p_Ptr + v_Size - 8), v_Byte); i != SIZE_MAX) return p_Ptr + v_Size - 8 + i;
				return nullptr;
			}
			if (v_Size >= 4) {
				if (const size_t i = findScalar(loadScalar<uint32_t>(p_Ptr), v_Byte); i != SIZE_MAX) return p_Ptr + i;
				if (const size_t i = findScalar(loadScalar<uint32_t>(p_Ptr + v_Size - 4), v_Byte); i != SIZE_MAX) return p_Ptr + v_Size - 4 + i;
				return nullptr;
			}
			if (v_Size == 0) return nullptr;
			if (p_Ptr[0] == v_Byte) return p_Ptr;
			if (p_Ptr[v_Size >> 1] == v_Byte) return p_Ptr + (v_Size >> 1);
			return p_Ptr[v_Size - 1] == v_Byte ? p_Ptr + v_Size - 1 : nullptr;
		}
	}

	bool memEqual(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size) {
		if (v_Size < 2 * V::VIntrospect<V::preferredBackend()>::kWidth)
			return smallEqual(static_cast<const uint8_t*>(p_Left), static_cast<const uint8_t*>(p_Right), v_Size);
		return Internal::memEqual<V::preferredBackend()>(p_Left, p_Right, v_Size);
	}

	int memCompare(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size) {
		if (v_Size < 2 * V::VIntrospect<V::preferredBackend()>::kWidth)
			return smallCompare(static_cast<const uint8_t*>(p_Left), static_cast<const uint8_t*>(p_Right), v_Size);
		return Internal::memCompare<V::preferredBackend()>(p_Left, p_Right, v_Size);
	}

	const void* memFindByte(const void* STL_RESTRICT p_Ptr, size_t v_Size, uint8_t v_Byte) {
		if (v_Size < 2 * V::VIntrospect<V::preferredBackend()>::kWidth)
			return smallFind(static_cast<const uint8_t*>(p_Ptr), v_Size, v_Byte);
		return Internal::memFindByte<V::preferredBackend()>(p_Ptr, v_Size, v_Byte);
	}

	void* memFindByte(void* STL_RESTRICT p_Ptr, size_t v_Size, uint8_t v_Byte) {
		return const_cast<void*>(memFindByte(static_cast<const void*>(p_Ptr), v_Size, v_Byte));
	}

	void prefetchRead(const void* STL_RESTRICT p_Ptr, int v_Locality) {
		switch (v_Locality) {
			case 0:
				Internal::PrefetchRead::invoke<0>(p_Ptr);
				break;
			case 1:
				Internal::PrefetchRead::invoke<1>(p_Ptr);
				break;
			case 2:
				Internal::PrefetchRead::invoke<2>(p_Ptr);
				break;
			case 3:
				Internal::PrefetchRead::invoke<3>(p_Ptr);
				break;
			default:
				break;
		}
	}

	void prefetchWrite(const void* STL_RESTRICT p_Ptr) {
		Internal::PrefetchWrite::invoke(p_Ptr);
	}
}