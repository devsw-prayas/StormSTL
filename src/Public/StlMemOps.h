#pragma once
#include "StlUnrolled.h"
#include "StormSTL.h"
#include "StlIntrin.h"
#include "StlVUtils.h"

namespace Stl::Memory {
	namespace Internal {
		using namespace Stl::Internal::V;

		template<VType Type>
		uint64_t byteEqualMask(const typename VIntrospect<Type>::template RType<uint8_t> v_Left, const typename VIntrospect<Type>::template RType<uint8_t> v_Right) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				return static_cast<uint64_t>(CompareEqual<Type, uint8_t>::invoke(v_Left, v_Right));
			} else
#endif
			{
				return MoveMask<Type, uint8_t>::invoke(CompareEqual<Type, uint8_t>::invoke(v_Left, v_Right));
			}
		}

		template<VType Type>
		bool memEqual(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size) {
			using r = VIntrospect<Type>::template RType<uint8_t>;
			const auto* currLeft = static_cast<const uint8_t*>(p_Left);
			const auto* currRight = static_cast<const uint8_t*>(p_Right);
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = VIntrospect<Type>::UType::value;
			constexpr size_t advance = width * window;
			constexpr uint64_t fullMask = width == 64 ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);

			for (size_t v = 0; v < v_Size; v += advance) {
				bool batchEqual = true;
				Unrolled::staticFor<0, window>([&](auto idx) {
					const r left = Load<Type, uint8_t>::invoke(currLeft + idx * width);
					const r right = Load<Type, uint8_t>::invoke(currRight + idx * width);
					batchEqual = batchEqual && (byteEqualMask<Type>(left, right) == fullMask);
				});
				if (!batchEqual) return false;
				currLeft += advance;
				currRight += advance;
			}
			return true;
		}

		template<VType Type>
		int memCompare(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size) {
			using r = VIntrospect<Type>::template RType<uint8_t>;
			const auto* currLeft = static_cast<const uint8_t*>(p_Left);
			const auto* currRight = static_cast<const uint8_t*>(p_Right);
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = VIntrospect<Type>::UType::value;
			constexpr size_t advance = width * window;
			constexpr uint64_t fullMask = width == 64 ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);

			for (size_t v = 0; v < v_Size; v += advance) {
				bool found = false;
				size_t mismatch = 0;
				Unrolled::staticFor<0, window>([&](auto idx) {
					if (found) return;
					const r left = Load<Type, uint8_t>::invoke(currLeft + idx * width);
					const r right = Load<Type, uint8_t>::invoke(currRight + idx * width);
					const uint64_t mismatchMask = (~byteEqualMask<Type>(left, right)) & fullMask;
					if (mismatchMask == 0) return;
					mismatch = idx * width;
					while ((mismatchMask & (uint64_t(1) << (mismatch - idx * width))) == 0)
						++mismatch;
					found = true;
				});
				if (found) return currLeft[mismatch] < currRight[mismatch] ? -1 : 1;
				currLeft += advance;
				currRight += advance;
			}
			return 0;
		}

		template<VType Type>
		const void* memFindByte(const void* STL_RESTRICT p_Ptr, size_t v_Size, uint8_t v_Byte) {
			using r = VIntrospect<Type>::template RType<uint8_t>;
			const auto* curr = static_cast<const uint8_t*>(p_Ptr);
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = VIntrospect<Type>::UType::value;
			constexpr size_t advance = width * window;
			constexpr uint64_t fullMask = width == 64 ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);
			const r needle = Set1<Type, uint8_t>::invoke(v_Byte);

			for (size_t v = 0; v < v_Size; v += advance) {
				bool found = false;
				size_t match = 0;
				Unrolled::staticFor<0, window>([&](auto idx) {
					if (found) return;
					const r values = Load<Type, uint8_t>::invoke(curr + idx * width);
					const uint64_t matchMask = byteEqualMask<Type>(values, needle) & fullMask;
					if (matchMask == 0) return;
					match = idx * width;
					while ((matchMask & (uint64_t(1) << (match - idx * width))) == 0)
						++match;
					found = true;
				});
				if (found) return curr + match;
				curr += advance;
			}
			return nullptr;
		}

#if STL_SSE_SUPPORT
		void STL_FORCEINLINE setMemorySSE(void* STL_RESTRICT p_Ptr, uint8_t v_Val, size_t v_Size) {
			// Optimized temporal mem-set
			using r = VIntrospect<VType::V_SSE>::RType<uint8_t>;
			const r m1 = Set1<VType::V_SSE, uint8_t>::invoke(v_Val);
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) { // Outer
#if STL_COMPILER_MSVC && STL_SSE_UNROLL_WINDOW == 4
				Store<VType::V_SSE, uint8_t>::invoke(curr, m1);
				Store<VType::V_SSE, uint8_t>::invoke(curr + 16, m1);
				Store<VType::V_SSE, uint8_t>::invoke(curr + 32, m1);
				Store<VType::V_SSE, uint8_t>::invoke(curr + 48, m1);
#else
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([&](auto idx) { Store<VType::V_SSE, uint8_t>::invoke(curr + idx * 16, m1); });
#endif
				curr += advance;
			}
		}
#endif

#if STL_AVX_SUPPORT
		void STL_FORCEINLINE setMemoryAVX(void* STL_RESTRICT p_Ptr, uint8_t v_Val, size_t v_Size) {
			// Optimized temporal mem-set
			using r = VIntrospect<VType::V_AVX>::RType<uint8_t>;
			const r m1 = Set1<VType::V_AVX, uint8_t>::invoke(v_Val);
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) { // Outer
#if STL_COMPILER_MSVC && STL_AVX_UNROLL_WINDOW == 4
				Store<VType::V_AVX, uint8_t>::invoke(curr, m1);
				Store<VType::V_AVX, uint8_t>::invoke(curr + 32, m1);
				Store<VType::V_AVX, uint8_t>::invoke(curr + 64, m1);
				Store<VType::V_AVX, uint8_t>::invoke(curr + 96, m1);
#else
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([&](auto idx) { Store<VType::V_AVX, uint8_t>::invoke(curr + idx * 32, m1); });
#endif
				curr += advance;
			}
		}
#endif

#if STL_AVX512_SUPPORT
		void STL_FORCEINLINE setMemoryAVX512(void* STL_RESTRICT p_Ptr, uint8_t v_Val, size_t v_Size) {
			// Optimized temporal mem-set
			using r = VIntrospect<VType::V_AVX512>::RType<uint8_t>;
			const r m1 = Set1<VType::V_AVX512, uint8_t>::invoke(v_Val);
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) { // Outer
#if STL_COMPILER_MSVC && STL_AVX512_UNROLL_WINDOW == 4
				Store<VType::V_AVX512, uint8_t>::invoke(curr, m1);
				Store<VType::V_AVX512, uint8_t>::invoke(curr + 64, m1);
				Store<VType::V_AVX512, uint8_t>::invoke(curr + 128, m1);
				Store<VType::V_AVX512, uint8_t>::invoke(curr + 192, m1);
#else
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([&](auto idx) { Store<VType::V_AVX512, uint8_t>::invoke(curr + idx * 64, m1); });
#endif
				curr += advance;
			}
		}

#endif

#if STL_SSE_SUPPORT
		void STL_FORCEINLINE streamMemorySSE(void* STL_RESTRICT p_Ptr, uint8_t v_Val, size_t v_Size) {
			using r = VIntrospect<VType::V_SSE>::RType<uint8_t>;
			const r m1 = Set1<VType::V_SSE, uint8_t>::invoke(v_Val);
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_SSE_UNROLL_WINDOW == 4
				Stream<VType::V_SSE, uint8_t>::invoke(curr, m1);
				Stream<VType::V_SSE, uint8_t>::invoke(curr + 16, m1);
				Stream<VType::V_SSE, uint8_t>::invoke(curr + 32, m1);
				Stream<VType::V_SSE, uint8_t>::invoke(curr + 48, m1);
#else
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([&](auto idx) { Stream<VType::V_SSE, uint8_t>::invoke(curr + idx * 16, m1); });
#endif
				curr += advance;
			}
		}
#endif

#if STL_AVX_SUPPORT
		void STL_FORCEINLINE streamMemoryAVX(void* STL_RESTRICT p_Ptr, uint8_t v_Val, size_t v_Size) {
			using r = VIntrospect<VType::V_AVX>::RType<uint8_t>;
			const r m1 = Set1<VType::V_AVX, uint8_t>::invoke(v_Val);
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX_UNROLL_WINDOW == 4
				Stream<VType::V_AVX, uint8_t>::invoke(curr, m1);
				Stream<VType::V_AVX, uint8_t>::invoke(curr + 32, m1);
				Stream<VType::V_AVX, uint8_t>::invoke(curr + 64, m1);
				Stream<VType::V_AVX, uint8_t>::invoke(curr + 96, m1);
#else
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([&](auto idx) { Stream<VType::V_AVX, uint8_t>::invoke(curr + idx * 32, m1); });
#endif
				curr += advance;
			}
		}
#endif

#if STL_AVX512_SUPPORT
		void STL_FORCEINLINE streamMemoryAVX512(void* STL_RESTRICT p_Ptr, uint8_t v_Val, size_t v_Size) {
			using r = VIntrospect<VType::V_AVX512>::RType<uint8_t>;
			const r m1 = Set1<VType::V_AVX512, uint8_t>::invoke(v_Val);
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX512_UNROLL_WINDOW == 4
				Stream<VType::V_AVX512, uint8_t>::invoke(curr, m1);
				Stream<VType::V_AVX512, uint8_t>::invoke(curr + 64, m1);
				Stream<VType::V_AVX512, uint8_t>::invoke(curr + 128, m1);
				Stream<VType::V_AVX512, uint8_t>::invoke(curr + 192, m1);
#else
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([&](auto idx) { Stream<VType::V_AVX512, uint8_t>::invoke(curr + idx * 64, m1); });
#endif
				curr += advance;
			}
		}
#endif

#if STL_SSE_SUPPORT
		void STL_FORCEINLINE setMemoryZeroSSE(void* STL_RESTRICT p_Ptr, size_t v_Size) {
			using r = VIntrospect<VType::V_SSE>::RType<uint8_t>;
			const r m1 = SetZero<VType::V_SSE, uint8_t>::invoke();
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_SSE_UNROLL_WINDOW == 4
				Store<VType::V_SSE, uint8_t>::invoke(curr, m1);
				Store<VType::V_SSE, uint8_t>::invoke(curr + 16, m1);
				Store<VType::V_SSE, uint8_t>::invoke(curr + 32, m1);
				Store<VType::V_SSE, uint8_t>::invoke(curr + 48, m1);
#else
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([&](auto idx) { Store<VType::V_SSE, uint8_t>::invoke(curr + idx * 16, m1); });
#endif
				curr += advance;
			}
		}
#endif

#if STL_AVX_SUPPORT
		void STL_FORCEINLINE setMemoryZeroAVX(void* STL_RESTRICT p_Ptr, size_t v_Size) {
			using r = VIntrospect<VType::V_AVX>::RType<uint8_t>;
			const r m1 = SetZero<VType::V_AVX, uint8_t>::invoke();
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX_UNROLL_WINDOW == 4
				Store<VType::V_AVX, uint8_t>::invoke(curr, m1);
				Store<VType::V_AVX, uint8_t>::invoke(curr + 32, m1);
				Store<VType::V_AVX, uint8_t>::invoke(curr + 64, m1);
				Store<VType::V_AVX, uint8_t>::invoke(curr + 96, m1);
#else
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([&](auto idx) { Store<VType::V_AVX, uint8_t>::invoke(curr + idx * 32, m1); });
#endif
				curr += advance;
			}
		}
#endif

#if STL_AVX512_SUPPORT
		void STL_FORCEINLINE setMemoryZeroAVX512(void* STL_RESTRICT p_Ptr, size_t v_Size) {
			using r = VIntrospect<VType::V_AVX512>::RType<uint8_t>;
			const r m1 = SetZero<VType::V_AVX512, uint8_t>::invoke();
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX512_UNROLL_WINDOW == 4
				Store<VType::V_AVX512, uint8_t>::invoke(curr, m1);
				Store<VType::V_AVX512, uint8_t>::invoke(curr + 64, m1);
				Store<VType::V_AVX512, uint8_t>::invoke(curr + 128, m1);
				Store<VType::V_AVX512, uint8_t>::invoke(curr + 192, m1);
#else
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([&](auto idx) { Store<VType::V_AVX512, uint8_t>::invoke(curr + idx * 64, m1); });
#endif
				curr += advance;
			}
		}
#endif

#if STL_SSE_SUPPORT
		void STL_FORCEINLINE copyMemorySSE(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_SSE_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_SSE, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 16);
				const auto v2 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 32);
				const auto v3 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 48);
				Store<VType::V_SSE, uint8_t>::invoke(currDst, v0);
				Store<VType::V_SSE, uint8_t>::invoke(currDst + 16, v1);
				Store<VType::V_SSE, uint8_t>::invoke(currDst + 32, v2);
				Store<VType::V_SSE, uint8_t>::invoke(currDst + 48, v3);
#else
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16);
					Store<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryUnalignedSSE(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_SSE_UNROLL_WINDOW == 4
				const auto v0 = Loadu<VType::V_SSE, uint8_t>::invoke(currSrc);
				const auto v1 = Loadu<VType::V_SSE, uint8_t>::invoke(currSrc + 16);
				const auto v2 = Loadu<VType::V_SSE, uint8_t>::invoke(currSrc + 32);
				const auto v3 = Loadu<VType::V_SSE, uint8_t>::invoke(currSrc + 48);
				Storeu<VType::V_SSE, uint8_t>::invoke(currDst, v0);
				Storeu<VType::V_SSE, uint8_t>::invoke(currDst + 16, v1);
				Storeu<VType::V_SSE, uint8_t>::invoke(currDst + 32, v2);
				Storeu<VType::V_SSE, uint8_t>::invoke(currDst + 48, v3);
#else
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Loadu<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16);
					Storeu<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryStreamSSE(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_SSE_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_SSE, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 16);
				const auto v2 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 32);
				const auto v3 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 48);
				Stream<VType::V_SSE, uint8_t>::invoke(currDst, v0);
				Stream<VType::V_SSE, uint8_t>::invoke(currDst + 16, v1);
				Stream<VType::V_SSE, uint8_t>::invoke(currDst + 32, v2);
				Stream<VType::V_SSE, uint8_t>::invoke(currDst + 48, v3);
#else
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16);
					Stream<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryReverseSSE(const void* p_Src, void* p_Dst, size_t v_Size) {
			if (v_Size == 0) return;
			auto currSrc = static_cast<const unsigned char*>(p_Src) + v_Size;
			auto currDst = static_cast<unsigned char*>(p_Dst) + v_Size;
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				currSrc -= advance;
				currDst -= advance;
#if STL_COMPILER_MSVC && STL_SSE_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_SSE, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 16);
				const auto v2 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 32);
				const auto v3 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 48);
				Store<VType::V_SSE, uint8_t>::invoke(currDst, v0);
				Store<VType::V_SSE, uint8_t>::invoke(currDst + 16, v1);
				Store<VType::V_SSE, uint8_t>::invoke(currDst + 32, v2);
				Store<VType::V_SSE, uint8_t>::invoke(currDst + 48, v3);
#else
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16);
					Store<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16, v_Reg);
				});
#endif
			}
		}

		void STL_FORCEINLINE copyMemoryStreamReverseSSE(const void* p_Src, void* p_Dst, size_t v_Size) {
			if (v_Size == 0) return;
			auto currSrc = static_cast<const unsigned char*>(p_Src) + v_Size;
			auto currDst = static_cast<unsigned char*>(p_Dst) + v_Size;
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				currSrc -= advance;
				currDst -= advance;
#if STL_COMPILER_MSVC && STL_SSE_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_SSE, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 16);
				const auto v2 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 32);
				const auto v3 = Load<VType::V_SSE, uint8_t>::invoke(currSrc + 48);
				Stream<VType::V_SSE, uint8_t>::invoke(currDst, v0);
				Stream<VType::V_SSE, uint8_t>::invoke(currDst + 16, v1);
				Stream<VType::V_SSE, uint8_t>::invoke(currDst + 32, v2);
				Stream<VType::V_SSE, uint8_t>::invoke(currDst + 48, v3);
#else
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16);
					Stream<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16, v_Reg);
				});
#endif
			}
		}
#endif

#if STL_AVX_SUPPORT
		void STL_FORCEINLINE copyMemoryAVX(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_AVX, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 32);
				const auto v2 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 64);
				const auto v3 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 96);
				Store<VType::V_AVX, uint8_t>::invoke(currDst, v0);
				Store<VType::V_AVX, uint8_t>::invoke(currDst + 32, v1);
				Store<VType::V_AVX, uint8_t>::invoke(currDst + 64, v2);
				Store<VType::V_AVX, uint8_t>::invoke(currDst + 96, v3);
#else
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32);
					Store<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryUnalignedAVX(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX_UNROLL_WINDOW == 4
				const auto v0 = Loadu<VType::V_AVX, uint8_t>::invoke(currSrc);
				const auto v1 = Loadu<VType::V_AVX, uint8_t>::invoke(currSrc + 32);
				const auto v2 = Loadu<VType::V_AVX, uint8_t>::invoke(currSrc + 64);
				const auto v3 = Loadu<VType::V_AVX, uint8_t>::invoke(currSrc + 96);
				Storeu<VType::V_AVX, uint8_t>::invoke(currDst, v0);
				Storeu<VType::V_AVX, uint8_t>::invoke(currDst + 32, v1);
				Storeu<VType::V_AVX, uint8_t>::invoke(currDst + 64, v2);
				Storeu<VType::V_AVX, uint8_t>::invoke(currDst + 96, v3);
#else
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Loadu<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32);
					Storeu<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryStreamAVX(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_AVX, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 32);
				const auto v2 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 64);
				const auto v3 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 96);
				Stream<VType::V_AVX, uint8_t>::invoke(currDst, v0);
				Stream<VType::V_AVX, uint8_t>::invoke(currDst + 32, v1);
				Stream<VType::V_AVX, uint8_t>::invoke(currDst + 64, v2);
				Stream<VType::V_AVX, uint8_t>::invoke(currDst + 96, v3);
#else
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32);
					Stream<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryReverseAVX(const void* p_Src, void* p_Dst, size_t v_Size) {
			if (v_Size == 0) return;
			auto currSrc = static_cast<const unsigned char*>(p_Src) + v_Size;
			auto currDst = static_cast<unsigned char*>(p_Dst) + v_Size;
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				currSrc -= advance;
				currDst -= advance;
#if STL_COMPILER_MSVC && STL_AVX_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_AVX, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 32);
				const auto v2 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 64);
				const auto v3 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 96);
				Store<VType::V_AVX, uint8_t>::invoke(currDst, v0);
				Store<VType::V_AVX, uint8_t>::invoke(currDst + 32, v1);
				Store<VType::V_AVX, uint8_t>::invoke(currDst + 64, v2);
				Store<VType::V_AVX, uint8_t>::invoke(currDst + 96, v3);
#else
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32);
					Store<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32, v_Reg);
				});
#endif
			}
		}

		void STL_FORCEINLINE copyMemoryStreamReverseAVX(const void* p_Src, void* p_Dst, size_t v_Size) {
			if (v_Size == 0) return;
			auto currSrc = static_cast<const unsigned char*>(p_Src) + v_Size;
			auto currDst = static_cast<unsigned char*>(p_Dst) + v_Size;
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				currSrc -= advance;
				currDst -= advance;
#if STL_COMPILER_MSVC && STL_AVX_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_AVX, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 32);
				const auto v2 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 64);
				const auto v3 = Load<VType::V_AVX, uint8_t>::invoke(currSrc + 96);
				Stream<VType::V_AVX, uint8_t>::invoke(currDst, v0);
				Stream<VType::V_AVX, uint8_t>::invoke(currDst + 32, v1);
				Stream<VType::V_AVX, uint8_t>::invoke(currDst + 64, v2);
				Stream<VType::V_AVX, uint8_t>::invoke(currDst + 96, v3);
#else
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32);
					Stream<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32, v_Reg);
				});
#endif
			}
		}
#endif

#if STL_AVX512_SUPPORT
		void STL_FORCEINLINE copyMemoryAVX512(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX512_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 64);
				const auto v2 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 128);
				const auto v3 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 192);
				Store<VType::V_AVX512, uint8_t>::invoke(currDst, v0);
				Store<VType::V_AVX512, uint8_t>::invoke(currDst + 64, v1);
				Store<VType::V_AVX512, uint8_t>::invoke(currDst + 128, v2);
				Store<VType::V_AVX512, uint8_t>::invoke(currDst + 192, v3);
#else
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64);
					Store<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryUnalignedAVX512(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX512_UNROLL_WINDOW == 4
				const auto v0 = Loadu<VType::V_AVX512, uint8_t>::invoke(currSrc);
				const auto v1 = Loadu<VType::V_AVX512, uint8_t>::invoke(currSrc + 64);
				const auto v2 = Loadu<VType::V_AVX512, uint8_t>::invoke(currSrc + 128);
				const auto v3 = Loadu<VType::V_AVX512, uint8_t>::invoke(currSrc + 192);
				Storeu<VType::V_AVX512, uint8_t>::invoke(currDst, v0);
				Storeu<VType::V_AVX512, uint8_t>::invoke(currDst + 64, v1);
				Storeu<VType::V_AVX512, uint8_t>::invoke(currDst + 128, v2);
				Storeu<VType::V_AVX512, uint8_t>::invoke(currDst + 192, v3);
#else
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Loadu<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64);
					Storeu<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryStreamAVX512(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
#if STL_COMPILER_MSVC && STL_AVX512_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 64);
				const auto v2 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 128);
				const auto v3 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 192);
				Stream<VType::V_AVX512, uint8_t>::invoke(currDst, v0);
				Stream<VType::V_AVX512, uint8_t>::invoke(currDst + 64, v1);
				Stream<VType::V_AVX512, uint8_t>::invoke(currDst + 128, v2);
				Stream<VType::V_AVX512, uint8_t>::invoke(currDst + 192, v3);
#else
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64);
					Stream<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64, v_Reg);
				});
#endif
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryReverseAVX512(const void* p_Src, void* p_Dst, size_t v_Size) {
			if (v_Size == 0) return;
			auto currSrc = static_cast<const unsigned char*>(p_Src) + v_Size;
			auto currDst = static_cast<unsigned char*>(p_Dst) + v_Size;
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				currSrc -= advance;
				currDst -= advance;
#if STL_COMPILER_MSVC && STL_AVX512_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 64);
				const auto v2 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 128);
				const auto v3 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 192);
				Store<VType::V_AVX512, uint8_t>::invoke(currDst, v0);
				Store<VType::V_AVX512, uint8_t>::invoke(currDst + 64, v1);
				Store<VType::V_AVX512, uint8_t>::invoke(currDst + 128, v2);
				Store<VType::V_AVX512, uint8_t>::invoke(currDst + 192, v3);
#else
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64);
					Store<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64, v_Reg);
				});
#endif
			}
		}

		void STL_FORCEINLINE copyMemoryStreamReverseAVX512(const void* p_Src, void* p_Dst, size_t v_Size) {
			if (v_Size == 0) return;
			auto currSrc = static_cast<const unsigned char*>(p_Src) + v_Size;
			auto currDst = static_cast<unsigned char*>(p_Dst) + v_Size;
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				currSrc -= advance;
				currDst -= advance;
#if STL_COMPILER_MSVC && STL_AVX512_UNROLL_WINDOW == 4
				const auto v0 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc);
				const auto v1 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 64);
				const auto v2 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 128);
				const auto v3 = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + 192);
				Stream<VType::V_AVX512, uint8_t>::invoke(currDst, v0);
				Stream<VType::V_AVX512, uint8_t>::invoke(currDst + 64, v1);
				Stream<VType::V_AVX512, uint8_t>::invoke(currDst + 128, v2);
				Stream<VType::V_AVX512, uint8_t>::invoke(currDst + 192, v3);
#else
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([&](auto idx) {
					const auto v_Reg = Load<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64);
					Stream<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64, v_Reg);
				});
#endif
			}
		}
#endif

		// Canonical Ops with assumptions.

		template<VType Type>
		void memSetDispatch(void* STL_RESTRICT p_Src, uint8_t v_Val, size_t v_Size) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				STL_ASSERT(v_Size % (STL_AVX512_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				setMemoryAVX512(p_Src, v_Val, v_Size);
			}
#endif
#if STL_AVX_SUPPORT
			if constexpr (Type == VType::V_AVX) {
				STL_ASSERT(v_Size % (STL_AVX_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				setMemoryAVX(p_Src, v_Val, v_Size);
			}
#endif
#if STL_SSE_SUPPORT
			if constexpr (Type == VType::V_SSE) {
				STL_ASSERT(v_Size % (STL_SSE_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				setMemorySSE(p_Src, v_Val, v_Size);
			}
#endif
		}

		template<VType Type>
		void memStreamDispatch(void* STL_RESTRICT p_Dst, uint8_t v_Val, size_t v_Size) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				STL_ASSERT(v_Size % (STL_AVX512_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				streamMemoryAVX512(p_Dst, v_Val, v_Size);
			}
#endif
#if STL_AVX_SUPPORT
			if constexpr (Type == VType::V_AVX) {
				STL_ASSERT(v_Size % (STL_AVX_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				streamMemoryAVX(p_Dst, v_Val, v_Size);
			}
#endif
#if STL_SSE_SUPPORT
			if constexpr (Type == VType::V_SSE) {
				STL_ASSERT(v_Size % (STL_SSE_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				streamMemorySSE(p_Dst, v_Val, v_Size);
			}
#endif
		}

		template<VType Type>
		void memZeroDispatch(void* STL_RESTRICT p_Src, size_t v_Size) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				STL_ASSERT(v_Size % (STL_AVX512_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				setMemoryZeroAVX512(p_Src, v_Size);
			}
#endif
#if STL_AVX_SUPPORT
			if constexpr (Type == VType::V_AVX) {
				STL_ASSERT(v_Size % (STL_AVX_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				setMemoryZeroAVX(p_Src, v_Size);
			}
#endif
#if STL_SSE_SUPPORT
			if constexpr (Type == VType::V_SSE) {
				STL_ASSERT(v_Size % (STL_SSE_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				setMemoryZeroSSE(p_Src, v_Size);
			}
#endif
		}

		template<VType Type>
		void memCopyDispatch(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				STL_ASSERT(v_Size % (STL_AVX512_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryAVX512(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_AVX_SUPPORT
			if constexpr (Type == VType::V_AVX) {
				STL_ASSERT(v_Size % (STL_AVX_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryAVX(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_SSE_SUPPORT
			if constexpr (Type == VType::V_SSE) {
				STL_ASSERT(v_Size % (STL_SSE_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemorySSE(p_Src, p_Dst, v_Size);
			}
#endif
		}

		template<VType Type>
		void memCopyUnalignedDispatch(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				STL_ASSERT(v_Size % (STL_AVX512_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryUnalignedAVX512(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_AVX_SUPPORT
			if constexpr (Type == VType::V_AVX) {
				STL_ASSERT(v_Size % (STL_AVX_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryUnalignedAVX(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_SSE_SUPPORT
			if constexpr (Type == VType::V_SSE) {
				STL_ASSERT(v_Size % (STL_SSE_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryUnalignedSSE(p_Src, p_Dst, v_Size);
			}
#endif
		}

		template<VType Type>
		void memCopyStreamDispatch(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				STL_ASSERT(v_Size % (STL_AVX512_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryStreamAVX512(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_AVX_SUPPORT
			if constexpr (Type == VType::V_AVX) {
				STL_ASSERT(v_Size % (STL_AVX_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryStreamAVX(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_SSE_SUPPORT
			if constexpr (Type == VType::V_SSE) {
				STL_ASSERT(v_Size % (STL_SSE_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryStreamSSE(p_Src, p_Dst, v_Size);
			}
#endif
		}

		template<VType Type>
		void memCopyReverseDispatch(const void* p_Src, void* p_Dst, size_t v_Size) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				STL_ASSERT(v_Size % (STL_AVX512_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryReverseAVX512(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_AVX_SUPPORT
			if constexpr (Type == VType::V_AVX) {
				STL_ASSERT(v_Size % (STL_AVX_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryReverseAVX(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_SSE_SUPPORT
			if constexpr (Type == VType::V_SSE) {
				STL_ASSERT(v_Size % (STL_SSE_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryReverseSSE(p_Src, p_Dst, v_Size);
			}
#endif
		}

		template<VType Type>
		void memCopyStreamReverseDispatch(const void* p_Src, void* p_Dst, size_t v_Size) {
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				STL_ASSERT(v_Size % (STL_AVX512_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryStreamReverseAVX512(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_AVX_SUPPORT
			if constexpr (Type == VType::V_AVX) {
				STL_ASSERT(v_Size % (STL_AVX_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryStreamReverseAVX(p_Src, p_Dst, v_Size);
			}
#endif
#if STL_SSE_SUPPORT
			if constexpr (Type == VType::V_SSE) {
				STL_ASSERT(v_Size % (STL_SSE_UNROLL_WINDOW * VIntrospect<Type>::kWidth) == 0);
				copyMemoryStreamReverseSSE(p_Src, p_Dst, v_Size);
			}
#endif
		}
	}

	STL_RUNTIME_API void memSet(void* STL_RESTRICT p_Dst, uint8_t v_Val, size_t v_Size);
	STL_RUNTIME_API void memCopy(void* STL_RESTRICT p_Dst, const void* STL_RESTRICT p_Src, size_t v_Size);
	STL_RUNTIME_API void memMove(void* p_Dst, const void* p_Src, size_t v_Size);
	STL_RUNTIME_API bool memEqual(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size);
	STL_RUNTIME_API int memCompare(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size);
	STL_RUNTIME_API void* memFindByte(void* STL_RESTRICT p_Ptr, size_t v_Size, uint8_t v_Byte);
	STL_RUNTIME_API const void* memFindByte(const void* STL_RESTRICT p_Ptr, size_t v_Size, uint8_t v_Byte);
	STL_RUNTIME_API void prefetchRead(const void* STL_RESTRICT p_Ptr, int v_Locality);
	STL_RUNTIME_API void prefetchWrite(const void* STL_RESTRICT p_Ptr);
}
