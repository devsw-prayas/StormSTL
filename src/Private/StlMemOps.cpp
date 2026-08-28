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

		auto* const s = reinterpret_cast<uint8_t*>((p + mask) & ~mask);
		auto* const e = reinterpret_cast<uint8_t*>((p + v_Size) & ~mask);
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
		constexpr auto align = introspect::kAlign - 1;
		const auto vSize = v_Size - (v_Size % (introspect::kWidth * introspect::UType::value));

		if ((reinterpret_cast<uintptr_t>(p_Dst) & align) != 0 || (reinterpret_cast<uintptr_t>(p_Src) & align) != 0) {
			Internal::memCopyUnalignedDispatch<V::preferredBackend()>(p_Src, p_Dst, vSize);
		} else if (vSize < nonTemporalThreshold()) {
			Internal::memCopyDispatch<V::preferredBackend()>(p_Src, p_Dst, vSize);
		} else {
			Internal::memCopyStreamDispatch<V::preferredBackend()>(p_Src, p_Dst, vSize);
			Internal::Fence<V::preferredBackend()>::invoke();
		}

		for (size_t i = vSize; i < v_Size; ++i)
			static_cast<uint8_t*>(p_Dst)[i] = static_cast<const uint8_t*>(p_Src)[i];
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