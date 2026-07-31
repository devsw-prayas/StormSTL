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
#include "StlDiagnostics.h"
#define ALLOW_SYSCALL

#include "StlSyscalls.h"
#include "StlMemoryInternal.h"

namespace Storm::STL::Memory {
	void NativeMemHandle::init() noexcept {
		m_BaseAddress = nullptr;
		m_TotalSize = 0;
		m_CommittedSize = 0;
	}

	void NativeMemHandle::setBaseAddress(void* p_BaseAddress) noexcept {
		m_BaseAddress = p_BaseAddress;
	}

	void NativeMemHandle::setTotalSize(size_t v_TotalSize) noexcept {
		m_TotalSize = v_TotalSize;
	}

	void NativeMemHandle::growCommitted(size_t v_Delta) noexcept {
		STL_ASSERT(m_CommittedSize + v_Delta <= m_TotalSize);
		m_CommittedSize += v_Delta;
	}

	void NativeMemHandle::shrinkCommitted(size_t v_Delta) noexcept {
		STL_ASSERT(v_Delta <= m_CommittedSize);
		m_CommittedSize -= v_Delta;
	}

	void NativeMemDesc::init() noexcept {
		m_Size = 0;
		m_Offset = 0;
		m_Protect = MemoryProtect::NO_ACCESS;
		m_Flags = MemoryFlags::NONE;
	}

	void NativeMemDesc::setSize(size_t v_Size) noexcept {
		m_Size = v_Size;
	}

	void NativeMemDesc::setOffset(size_t v_Offset) noexcept {
		m_Offset = v_Offset;
	}

	void NativeMemDesc::setProtect(MemoryProtect v_Protect) noexcept {
		m_Protect = v_Protect;
	}

	void NativeMemDesc::setFlags(MemoryFlags v_Flags) noexcept {
		m_Flags = v_Flags;
	}
}

namespace Storm::STL::Memory::Internal {
#if defined(_WIN32)
	DWORD toWin32Protect(MemoryProtect v_Protect) {
		switch (v_Protect) {
		case MemoryProtect::NO_ACCESS:          return PAGE_NOACCESS;
		case MemoryProtect::READ_ONLY:          return PAGE_READONLY;
		case MemoryProtect::READ_WRITE:         return PAGE_READWRITE;
		case MemoryProtect::EXECUTE:             return PAGE_EXECUTE;
		case MemoryProtect::EXECUTE_READ:        return PAGE_EXECUTE_READ;
		case MemoryProtect::EXECUTE_READ_WRITE:  return PAGE_EXECUTE_READWRITE;
		case MemoryProtect::GUARD:               return PAGE_READWRITE | PAGE_GUARD;
		}
		STL_ASSERT(false);
		return PAGE_NOACCESS;
	}
#elif defined(__linux__)
	int toPosixProtect(MemoryProtect v_Protect) {
		switch (v_Protect) {
		case MemoryProtect::NO_ACCESS:          return PROT_NONE;
		case MemoryProtect::READ_ONLY:          return PROT_READ;
		case MemoryProtect::READ_WRITE:         return PROT_READ | PROT_WRITE;
		case MemoryProtect::EXECUTE:             return PROT_EXEC;
		case MemoryProtect::EXECUTE_READ:        return PROT_EXEC | PROT_READ;
		case MemoryProtect::EXECUTE_READ_WRITE:  return PROT_EXEC | PROT_READ | PROT_WRITE;
		case MemoryProtect::GUARD:               return PROT_NONE;
		}
		STL_ASSERT(false);
		return PROT_NONE;
	}
#endif

	void releaseRaw(void* p_Base, size_t v_TotalSize) {
		if (p_Base == nullptr) return;
#if defined(_WIN32)
		STL_UNUSED(v_TotalSize);
		VirtualFree(p_Base, 0, MEM_RELEASE);
#elif defined(__linux__)
		munmap(p_Base, v_TotalSize);
#endif
	}

	const NativePageInfo& pageInfo() {
		static const NativePageInfo s_PageInfo = [] {
			NativePageInfo info{};
#if defined(_WIN32)
			SYSTEM_INFO sysInfo{};
			GetSystemInfo(&sysInfo);
			info.m_PageSize = sysInfo.dwPageSize;
			info.m_AllocationGranularity = sysInfo.dwAllocationGranularity;
#elif defined(__linux__)
			const long pageSize = sysconf(_SC_PAGESIZE);
			info.m_PageSize = pageSize > 0 ? static_cast<size_t>(pageSize) : 4096;
			info.m_AllocationGranularity = info.m_PageSize;
#else
			info.m_PageSize = 4096;
			info.m_AllocationGranularity = 4096;
#endif
			return info;
		}();
		return s_PageInfo;
	}
}
