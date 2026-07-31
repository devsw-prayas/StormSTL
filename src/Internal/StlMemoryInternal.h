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
#pragma once

#include "StlMemoryUtils.h"

namespace Storm::STL::Memory::Internal {
	struct NativePageInfo final {
		size_t m_PageSize;
		size_t m_AllocationGranularity;
	};

	const NativePageInfo& pageInfo();

	void releaseRaw(void* p_Base, size_t v_TotalSize);

#if defined(_WIN32)
	DWORD toWin32Protect(MemoryProtect v_Protect);
#elif defined(__linux__)
	int toPosixProtect(MemoryProtect v_Protect);
#endif
}
