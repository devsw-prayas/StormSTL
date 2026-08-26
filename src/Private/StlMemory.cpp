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
		const size_t alignedSize = alignToGranularity(ro_Desc.m_Size);

#if defined(_WIN32)
		DWORD allocType = MEM_RESERVE;
		if (ro_Desc.m_Flags == MemoryFlags::LARGE_PAGES) {
			allocType |= MEM_LARGE_PAGES;
		}
		void* base = VirtualAlloc(nullptr, alignedSize, allocType, PAGE_NOACCESS);
#elif defined(__linux__)
		int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#ifdef MAP_HUGETLB
		if (ro_Desc.m_Flags == MemoryFlags::LARGE_PAGES) {
			flags |= MAP_HUGETLB;
		}
#endif
		void* mapped = mmap(nullptr, alignedSize, PROT_NONE, flags, -1, 0);
		void* base = (mapped == MAP_FAILED) ? nullptr : mapped;
#else
		void* base = nullptr;
#endif

		if (base == nullptr) {
			STL_ASSERT(false);
			return INVALID_NATIVE_HANDLE;
		}

		NativeMemHandle handle{};
		handle.init();
		handle.setBaseAddress(base);
		handle.setTotalSize(alignedSize);
		return handle;
	}

	void NativeMemory::commitMore(NativeMemHandle& r_Handle, const NativeMemDesc& ro_Desc) {
		STL_ASSERT(r_Handle.m_BaseAddress != nullptr);
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
