/*
* Copyright (c) 2026 StormWeaver
*
* This file is part of the StormSTL (Standard Template Library)
*
* Licensed under the MIT License. You may obtain a copy of the License at
* https://opensource.org/licenses/MIT
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
*/
#include "StormSTL.h"
#include "StlMemory.h"
#include "StlDiagnostics.h"
#define ALLOW_SYSCALL

#include "StlSyscalls.h"
#include "StlMemoryInternal.h"

namespace Storm::STL::Memory {
	using namespace Internal;
	
	size_t NativeMemory::alignToPage(size_t v_Size) {
		const size_t page = pageInfo().m_PageSize;
		return (v_Size + page - 1) & ~(page - 1);
	}

	size_t NativeMemory::alignToGranularity(size_t v_Size) {
		const size_t granularity = pageInfo().m_AllocationGranularity;
		return (v_Size + granularity - 1) & ~(granularity - 1);
	}

	bool NativeMemory::isPageAligned(size_t v_Size) {
		return (v_Size & (pageInfo().m_PageSize - 1)) == 0;
	}

	bool NativeMemory::isGranularityAligned(size_t v_Size) {
		return (v_Size & (pageInfo().m_AllocationGranularity - 1)) == 0;
	}

	void* NativeMemory::alignPointerToPage(void* p_MemoryAddr) {
		return reinterpret_cast<void*>(alignToPage(reinterpret_cast<uintptr_t>(p_MemoryAddr)));
	}

	void* NativeMemory::alignPointerToGranularity(void* p_MemoryAddr) {
		return reinterpret_cast<void*>(alignToGranularity(reinterpret_cast<uintptr_t>(p_MemoryAddr)));
	}

	bool NativeMemory::isPointerPageAligned(void* p_MemoryAddr) {
		return isPageAligned(reinterpret_cast<uintptr_t>(p_MemoryAddr));
	}

	bool NativeMemory::isPointerGranularityAligned(void* p_MemoryAddr) {
		return isGranularityAligned(reinterpret_cast<uintptr_t>(p_MemoryAddr));
	}

	NativeMemHandle NativeMemory::reserve(const NativeMemDesc& ro_Desc) {
		STL_ASSERT(ro_Desc.m_Size > 0);
		const bool large = ro_Desc.m_Flags == MemoryFlags::LARGE_PAGES;
		const size_t largeSize = large ? largePageSize() : 0;

		// Large pages are best-effort (missing privilege, no THP, fragmentation): fail quietly so the caller can retry without.
		if (large && largeSize == 0) return INVALID_NATIVE_HANDLE;

		const size_t alignedSize = large ? (ro_Desc.m_Size + largeSize - 1) & ~(largeSize - 1) : alignToGranularity(ro_Desc.m_Size);
		void* base = nullptr;

#if defined(_WIN32)
		if (large) {
			// MEM_LARGE_PAGES is only valid together with MEM_COMMIT, so the region is committed whole and never grows or shrinks.
			if (!enableLockMemoryPrivilege()) return INVALID_NATIVE_HANDLE;
			const DWORD protect = ro_Desc.m_Protect == MemoryProtect::NO_ACCESS ? PAGE_READWRITE : toWin32Protect(ro_Desc.m_Protect);
			base = VirtualAlloc(nullptr, alignedSize, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, protect);
		} else {
			base = VirtualAlloc(nullptr, alignedSize, MEM_RESERVE, PAGE_NOACCESS);
		}
#elif defined(__linux__)
		constexpr int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
		if (large) {
			// Transparent huge pages only back 2 MiB-aligned ranges, so over-map and trim to an aligned window.
			const size_t mapSize = alignedSize + largeSize;
			void* mapped = mmap(nullptr, mapSize, PROT_NONE, flags, -1, 0);
			if (mapped == MAP_FAILED) return INVALID_NATIVE_HANDLE;

			const uintptr_t raw = reinterpret_cast<uintptr_t>(mapped);
			const uintptr_t aligned = (raw + largeSize - 1) & ~(largeSize - 1);
			const size_t head = aligned - raw;
			const size_t tail = mapSize - head - alignedSize;
			if (head != 0) munmap(mapped, head);
			if (tail != 0) munmap(reinterpret_cast<void*>(aligned + alignedSize), tail);

			base = reinterpret_cast<void*>(aligned);
#ifdef MADV_HUGEPAGE
			madvise(base, alignedSize, MADV_HUGEPAGE);
#endif
		} else {
			void* mapped = mmap(nullptr, alignedSize, PROT_NONE, flags, -1, 0);
			base = (mapped == MAP_FAILED) ? nullptr : mapped;
		}
#endif

		if (base == nullptr) {
			STL_ASSERT(large);
			return INVALID_NATIVE_HANDLE;
		}

		NativeMemHandle handle{};
		handle.init();
		handle.setBaseAddress(base);
		handle.setTotalSize(alignedSize);
		handle.setFlags(ro_Desc.m_Flags);
#if defined(_WIN32)
		if (large) handle.growCommitted(alignedSize);
#endif
		return handle;
	}

	void NativeMemory::commitMore(NativeMemHandle& r_Handle, const NativeMemDesc& ro_Desc) {
		STL_ASSERT(r_Handle.m_BaseAddress != nullptr);
#if defined(_WIN32)
		// Large-page regions are committed whole at reserve; nothing is left to commit.
		if (r_Handle.m_Flags == MemoryFlags::LARGE_PAGES) return;
#endif
		const size_t alignedSize = alignToPage(ro_Desc.m_Size);

		void* target = static_cast<uint8_t*>(r_Handle.m_BaseAddress) + r_Handle.m_CommittedSize;

#if defined(_WIN32)
		void* result = VirtualAlloc(target, alignedSize, MEM_COMMIT, toWin32Protect(ro_Desc.m_Protect));
		if (result != target) {
			STL_ASSERT(false);
			releaseRaw(r_Handle.m_BaseAddress, r_Handle.m_TotalSize);
			r_Handle = INVALID_NATIVE_HANDLE;
			return;
		}
#elif defined(__linux__)
		int result = mprotect(target, alignedSize, toPosixProtect(ro_Desc.m_Protect));
		if (result != 0) {
			STL_ASSERT(false);
			releaseRaw(r_Handle.m_BaseAddress, r_Handle.m_TotalSize);
			r_Handle = INVALID_NATIVE_HANDLE;
			return;
		}
#endif

		r_Handle.growCommitted(alignedSize);
	}

	void NativeMemory::decommitTail(NativeMemHandle& r_Handle, const NativeMemDesc& ro_Desc) {
		STL_ASSERT(r_Handle.m_BaseAddress != nullptr);
#if defined(_WIN32)
		// Windows large pages cannot be decommitted in part.
		STL_ASSERT(r_Handle.m_Flags != MemoryFlags::LARGE_PAGES);
#endif
		const size_t alignedSize = alignToPage(ro_Desc.m_Size);
		STL_ASSERT(alignedSize <= r_Handle.m_CommittedSize);

		void* target = static_cast<uint8_t*>(r_Handle.m_BaseAddress) + (r_Handle.m_CommittedSize - alignedSize);

#if defined(_WIN32)
		BOOL ok = VirtualFree(target, alignedSize, MEM_DECOMMIT);
		STL_ASSERT(ok != 0);
#elif defined(__linux__)
		int ok = madvise(target, alignedSize, MADV_DONTNEED);
		STL_ASSERT(ok == 0);
		ok = mprotect(target, alignedSize, PROT_NONE);
		STL_ASSERT(ok == 0);
#endif

		r_Handle.shrinkCommitted(alignedSize);
	}

	void NativeMemory::protect(NativeMemHandle& r_Handle, const NativeMemDesc& ro_Desc) {
		STL_ASSERT(r_Handle.m_BaseAddress != nullptr);
		STL_ASSERT(ro_Desc.m_Offset + ro_Desc.m_Size <= r_Handle.m_CommittedSize);

		void* target = static_cast<uint8_t*>(r_Handle.m_BaseAddress) + ro_Desc.m_Offset;

#if defined(_WIN32)
		DWORD oldProtect = 0;
		BOOL ok = VirtualProtect(target, ro_Desc.m_Size, toWin32Protect(ro_Desc.m_Protect), &oldProtect);
		STL_ASSERT(ok != 0);
#elif defined(__linux__)
		int ok = mprotect(target, ro_Desc.m_Size, toPosixProtect(ro_Desc.m_Protect));
		STL_ASSERT(ok == 0);
#endif
	}

	void NativeMemory::release(NativeMemHandle& r_Handle) {
		releaseRaw(r_Handle.m_BaseAddress, r_Handle.m_TotalSize);

		r_Handle = INVALID_NATIVE_HANDLE;
	}
}
