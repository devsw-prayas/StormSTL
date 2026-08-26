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
#include "StlCompiler.h"
#include "StlDiagnostics.h"
#include <type_traits>
#include <utility>
#include <new>

namespace Storm::STL::Memory {
	struct NativeMemHandle;
}

namespace Storm::STL::Memory::Allocators {
	struct AllocatorTrackingBase {
	protected:
		void track() noexcept {}
	};

	// Tier 0 - IArena: raw void*, bump/watermark only, no individual free.

	template<typename D>
	concept ArenaImpl = requires(D & d, size_t n, size_t a) {
		{ d.allocateImpl(n, a) } -> std::same_as<void*>;
		{ d.allocateImpl(n) } -> std::same_as<void*>;
		{ d.deallocateImpl() } -> std::same_as<void>;
	};

	template<typename D>
	class IArena : public AllocatorTrackingBase {
		using derived_ = D;

	public:
		IArena() noexcept {
			STL_STATIC_ASSERT(ArenaImpl<D>,
							   "Arena must implement allocateImpl(bytes,align)->void*, allocateImpl(bytes)->void*, deallocateImpl()->void");
		}

		IArena(const IArena&) = default;
		IArena& operator=(const IArena&) = default;
		IArena(IArena&&) noexcept = default;
		IArena& operator=(IArena&&) noexcept = default;
		~IArena() = default;

		STL_NODISCARD_MSG("Cannot discard allocated block pointer")
			void* allocate(size_t v_Bytes, size_t v_Alignment) {
			return static_cast<derived_*>(this)->allocateImpl(v_Bytes, v_Alignment);
		}

		STL_NODISCARD_MSG("Cannot discard allocated block pointer")
			void* allocate(size_t v_Bytes) {
			return static_cast<derived_*>(this)->allocateImpl(v_Bytes);
		}

		// Whole-arena discard only - no per-pointer free at this tier.
		void reset() {
			static_cast<derived_*>(this)->deallocateImpl();
		}
	};

	// Tier 1 - IAllocator<T, Arena, D>: consumes an Arena, presents contiguous
	// array behaviour over T. T = void is the raw/untyped specialization.

	template<typename D>
	concept RawAllocatorImpl = requires(D & d, void* p, size_t n, size_t a) {
		{ d.allocateImpl(n, a) } -> std::same_as<void*>;
		{ d.allocateImpl(n) } -> std::same_as<void*>;
		{ d.deallocateImpl(p, n) } -> std::same_as<void>;
	};

	template<typename D, typename T>
	concept TypedAllocatorImpl = requires(D & d, T* p, size_t n) {
		{ d.allocateImpl(n) } -> std::same_as<T*>;
		{ d.deallocateImpl(p, n) } -> std::same_as<void>;
	};

	template<typename T, typename Arena, typename D>
	class IAllocator : public AllocatorTrackingBase {
		using derived_ = D;

	protected:
		Arena m_UnderlyingArena;

	public:
		IAllocator() noexcept {
			if constexpr (std::is_void_v<T>) {
				STL_STATIC_ASSERT(RawAllocatorImpl<D>,
								   "Raw allocator must implement allocateImpl(bytes,align)->void*, allocateImpl(bytes)->void*, deallocateImpl(void*,size_t)->void");
			} else {
				STL_STATIC_ASSERT((TypedAllocatorImpl<D, T>),
								   "Typed allocator must implement allocateImpl(count)->T*, deallocateImpl(T*,size_t)->void");
			}
		}

		IAllocator(const IAllocator&) = default;
		IAllocator& operator=(const IAllocator&) = default;
		IAllocator(IAllocator&&) noexcept = default;
		IAllocator& operator=(IAllocator&&) noexcept = default;
		~IAllocator() = default;

		void init(NativeMemHandle* p_Handle) noexcept {
			m_UnderlyingArena.init(p_Handle);
		}

		// ---- Raw (T = void) ----
		STL_NODISCARD_MSG("Cannot discard allocated block pointer")
			void* allocate(size_t v_Bytes, size_t v_Alignment) requires std::is_void_v<T> {
			return static_cast<derived_*>(this)->allocateImpl(v_Bytes, v_Alignment);
		}

		STL_NODISCARD_MSG("Cannot discard allocated block pointer")
			void* allocate(size_t v_Bytes) requires std::is_void_v<T> {
			return static_cast<derived_*>(this)->allocateImpl(v_Bytes);
		}

		// Placement-new at a caller-supplied, already-allocated pointer only.
		template<typename U, typename... Args>
		U* emplace(void* p_Ptr, Args&&... v_Args) requires std::is_void_v<T> {
			STL_ASSERT(p_Ptr != nullptr);
			return ::new (p_Ptr) U(std::forward<Args>(v_Args)...);
		}

		// Policy-defined: no-op for pure-bump derivations, real freelist reclaim
		// for header/freelist-driven derivations
		void deallocate(void* p_Ptr, size_t v_Size) requires std::is_void_v<T> {
			static_cast<derived_*>(this)->deallocateImpl(p_Ptr, v_Size);
		}

		// Whole-arena discard. Derived types that track extra state on top of the Arena
		void reset() requires std::is_void_v<T> {
			m_UnderlyingArena.reset();
		}

		// Pure-bump defaults: forward straight to the Arena, no individual reclaim.
		void* allocateImpl(size_t v_Bytes, size_t v_Alignment) requires std::is_void_v<T> {
			return m_UnderlyingArena.allocate(v_Bytes, v_Alignment);
		}

		void* allocateImpl(size_t v_Bytes) requires std::is_void_v<T> {
			return m_UnderlyingArena.allocate(v_Bytes);
		}

		void deallocateImpl(void*, size_t) noexcept requires std::is_void_v<T> {}

		// ---- Typed (T != void) ----
		// Forward scaffolding only - not consumed by any concrete allocator yet.

		STL_NODISCARD_MSG("Cannot discard allocated block pointer")
			T* allocate(size_t v_Count = 1) requires (!std::is_void_v<T>) {
			return static_cast<derived_*>(this)->allocateImpl(v_Count);
		}

		// Pure-bump default: forwards straight to the Arena, same role as the Raw
		// tier's allocateImpl default.
		T* allocateImpl(size_t v_Count) requires (!std::is_void_v<T>) {
			return static_cast<T*>(m_UnderlyingArena.allocate(v_Count * sizeof(T), alignof(T)));
		}

		// Placement-new at a caller-supplied, already-allocated pointer only.
		template<typename... Args>
		T* emplace(T* p_Ptr, Args&&... v_Args) requires (!std::is_void_v<T>) {
			STL_ASSERT(p_Ptr != nullptr);
			return ::new (p_Ptr) T(std::forward<Args>(v_Args)...);
		}

		void destroy(T* p_Ptr, size_t v_Count = 1) requires (!std::is_void_v<T>) {
			for (size_t i = 0; i < v_Count; ++i)
				p_Ptr[i].~T();
		}

		void deallocate(T* p_Ptr, size_t v_Count = 1) requires (!std::is_void_v<T>) {
			static_cast<derived_*>(this)->deallocateImpl(p_Ptr, v_Count);
		}

		void deallocateImpl(T*, size_t) noexcept requires (!std::is_void_v<T>) {}
	};
}
