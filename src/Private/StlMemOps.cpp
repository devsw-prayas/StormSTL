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

	bool memEqual(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size) {
		constexpr size_t width = V::VIntrospect<V::preferredBackend()>::kWidth;
		if (v_Size < width) {
			const auto* a = static_cast<const uint8_t*>(p_Left);
			const auto* b = static_cast<const uint8_t*>(p_Right);
			for (size_t i = 0; i < v_Size; ++i)
				if (a[i] != b[i]) return false;
			return true;
		}
		return Internal::memEqual<V::preferredBackend()>(p_Left, p_Right, v_Size);
	}

	int memCompare(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size) {
		constexpr size_t width = V::VIntrospect<V::preferredBackend()>::kWidth;
		if (v_Size < width) {
			const auto* a = static_cast<const uint8_t*>(p_Left);
			const auto* b = static_cast<const uint8_t*>(p_Right);
			for (size_t i = 0; i < v_Size; ++i)
				if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
			return 0;
		}
		return Internal::memCompare<V::preferredBackend()>(p_Left, p_Right, v_Size);
	}

	const void* memFindByte(const void* STL_RESTRICT p_Ptr, size_t v_Size, uint8_t v_Byte) {
		constexpr size_t width = V::VIntrospect<V::preferredBackend()>::kWidth;
		if (v_Size < width) {
			const auto* p = static_cast<const uint8_t*>(p_Ptr);
			for (size_t i = 0; i < v_Size; ++i)
				if (p[i] == v_Byte) return p + i;
			return nullptr;
		}
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