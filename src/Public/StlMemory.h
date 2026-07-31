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

#include "StormSTL.h"
#include "StlMemoryUtils.h"

namespace Storm::STL::Memory {
	class STL_RUNTIME_API NativeMemory final {
	public:
		static size_t alignToPage(size_t v_Size);
		static size_t alignToGranularity(size_t v_Size);
		static bool isPageAligned(size_t v_Size);
		static bool isGranularityAligned(size_t v_Size);

		static void* alignPointerToPage(void* p_MemoryAddr);
		static void* alignPointerToGranularity(void* p_MemoryAddr);
		static bool isPointerPageAligned(void* p_MemoryAddr);
		static bool isPointerGranularityAligned(void* p_MemoryAddr);

		static NativeMemHandle reserve(const NativeMemDesc& ro_Desc);
		static void commitMore(NativeMemHandle& r_Handle, const NativeMemDesc& ro_Desc);
		static void decommitTail(NativeMemHandle& r_Handle, const NativeMemDesc& ro_Desc);
		static void protect(NativeMemHandle& r_Handle, const NativeMemDesc& ro_Desc);
		static void release(NativeMemHandle& r_Handle);
	};
}
