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

		if (v_Size < W || v_Size <= stosThreshold()) {
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
		const bool nt = v_Size > nonTemporalThreshold();
		for (auto* q = s; q < e; q += W) {
			if (nt) Internal::Stream<V::preferredBackend(), uint8_t>::invoke(q, l);
			else Internal::Store<V::preferredBackend(), uint8_t>::invoke(q, l);
		}
		if (nt) Internal::Fence<V::preferredBackend()>::invoke();

		Internal::Storeu<V::preferredBackend(), uint8_t>::invoke(d + v_Size - W, l);
	}

	void memCopy(void* STL_RESTRICT p_Dst, const void* STL_RESTRICT p_Src, size_t v_Size) {
		using introspect = V::VIntrospect<V::preferredBackend()>;
		constexpr size_t advance = introspect::kWidth * introspect::UType::value;
		constexpr uintptr_t align = introspect::kAlign - 1;

		if (v_Size == 0) return;

		// Sub-window sizes: ERMS
		if (v_Size < advance) {
#if STL_COMPILER_MSVC
			__movsb(static_cast<uint8_t*>(p_Dst), static_cast<const uint8_t*>(p_Src), v_Size);
#else
			__builtin_memcpy(p_Dst, p_Src, v_Size);
#endif
			return;
		}

		const size_t bulk = v_Size - (v_Size % advance);
		const bool unaligned = (reinterpret_cast<uintptr_t>(p_Dst) & align) != 0
			|| (reinterpret_cast<uintptr_t>(p_Src) & align) != 0;

		if (unaligned) {
			Internal::memCopyUnalignedDispatch<V::preferredBackend()>(p_Src, p_Dst, bulk);
		} else if (bulk < nonTemporalThreshold()) {
			Internal::memCopyDispatch<V::preferredBackend()>(p_Src, p_Dst, bulk);
		} else {
			Internal::memCopyStreamDispatch<V::preferredBackend()>(p_Src, p_Dst, bulk);
			Internal::Fence<V::preferredBackend()>::invoke();
		}

		// Tail: one unaligned window at the buffer end. Overlap is idempotent under __restrict.
		if (bulk != v_Size) {
			const size_t off = v_Size - advance;
			Internal::memCopyUnalignedDispatch<V::preferredBackend()>(
				static_cast<const uint8_t*>(p_Src) + off,
				static_cast<uint8_t*>(p_Dst) + off, advance);
		}
	}

	// TODO: implement.
	void memMove(void*, const void*, size_t) {}
	bool memEqual(const void* STL_RESTRICT, const void* STL_RESTRICT, size_t) {
		return false;
	}
	int memCompare(const void* STL_RESTRICT, const void* STL_RESTRICT, size_t) {
		return 0;
	}
	void* memFindByte(void* STL_RESTRICT, size_t, uint8_t) {
		return nullptr;
	}
	const void* memFindByte(const void* STL_RESTRICT, size_t, uint8_t) {
		return nullptr;
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