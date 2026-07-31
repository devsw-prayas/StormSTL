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

namespace Storm::STL::Memory {
	enum class STL_RUNTIME_API MemoryState : uint8_t {
		UNINITIALIZED, RESERVE, COMMIT, DECOMMIT, RELEASE, PROTECT
	};

	enum class STL_RUNTIME_API MemoryProtect : uint8_t {
		NO_ACCESS, READ_ONLY, READ_WRITE, EXECUTE, EXECUTE_READ, EXECUTE_READ_WRITE, GUARD
	};

	enum class STL_RUNTIME_API MemoryFlags : uint8_t {
		NONE, LARGE_PAGES
	};

	struct STL_ALIGNAS(32) STL_RUNTIME_API NativeMemHandle final {
		void* m_BaseAddress;
		size_t m_TotalSize;
		size_t m_CommittedSize;

		NativeMemHandle() = default;
		~NativeMemHandle() = default;

		NativeMemHandle(const NativeMemHandle&) = default;
		NativeMemHandle& operator=(const NativeMemHandle&) = default;

		NativeMemHandle(NativeMemHandle&&) noexcept = default;
		NativeMemHandle& operator=(NativeMemHandle&&) noexcept = default;

		void init() noexcept;

		void setBaseAddress(void* p_BaseAddress) noexcept;
		void setTotalSize(size_t v_TotalSize) noexcept;
		void growCommitted(size_t v_Delta) noexcept;
		void shrinkCommitted(size_t v_Delta) noexcept;
	};

	constexpr NativeMemHandle INVALID_NATIVE_HANDLE{};

	struct STL_ALIGNAS(32) STL_RUNTIME_API NativeMemDesc final {
		size_t m_Size;
		size_t m_Offset;
		MemoryProtect m_Protect;
		MemoryFlags m_Flags;

		NativeMemDesc() = default;
		~NativeMemDesc() = default;

		NativeMemDesc(const NativeMemDesc&) = default;
		NativeMemDesc& operator=(const NativeMemDesc&) = default;

		NativeMemDesc(NativeMemDesc&&) noexcept = default;
		NativeMemDesc& operator=(NativeMemDesc&&) noexcept = default;

		void init() noexcept;

		void setSize(size_t v_Size) noexcept;
		void setOffset(size_t v_Offset) noexcept;
		void setProtect(MemoryProtect v_Protect) noexcept;
		void setFlags(MemoryFlags v_Flags) noexcept;
	};
}
