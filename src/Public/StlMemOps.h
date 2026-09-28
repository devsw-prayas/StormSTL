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

		// Scanners keep a whole window of lanes live plus an accumulator and a constant; above 8
		// lanes that overflows the 16 xmm/ymm registers and spills every window (SSE at 16: ~8x slower).
		template<VType Type>
		consteval size_t scanWindow() { return VIntrospect<Type>::UType::value < 8 ? VIntrospect<Type>::UType::value : 8; }

		// The three scanners below assume v_Size >= kWidth. Each window is reduced in-register
		// (OR of per-lane XOR / compare results) so the hot loop pays one test per window, and
		// the per-lane masks are only extracted once a window is known to hit.

		template<VType Type>
		bool memEqual(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size) {
			const auto* l = static_cast<const uint8_t*>(p_Left);
			const auto* rp = static_cast<const uint8_t*>(p_Right);
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = scanWindow<Type>();
			constexpr size_t advance = width * window;

			const auto diff = [l, rp](size_t o) {
				return BitXor<Type>::invoke(Loadu<Type, uint8_t>::invoke(l + o), Loadu<Type, uint8_t>::invoke(rp + o));
			};

			size_t v = 0;
			for (; v + advance <= v_Size; v += advance) {
				typename VIntrospect<Type>::template RType<uint8_t> lanes[window];
				Unrolled::staticFor<0, window>([&](auto idx) { lanes[idx] = diff(v + idx * width); });
				auto acc = lanes[0];
				Unrolled::staticFor<1, window>([&](auto idx) { acc = BitOr<Type>::invoke(acc, lanes[idx]); });
				if (!TestZero<Type>::invoke(acc)) return false;
			}
			for (; v + width <= v_Size; v += width)
				if (!TestZero<Type>::invoke(diff(v))) return false;
			return v == v_Size || TestZero<Type>::invoke(diff(v_Size - width));
		}

		template<VType Type>
		int memCompare(const void* STL_RESTRICT p_Left, const void* STL_RESTRICT p_Right, size_t v_Size) {
			using r = VIntrospect<Type>::template RType<uint8_t>;
			const auto* l = static_cast<const uint8_t*>(p_Left);
			const auto* rp = static_cast<const uint8_t*>(p_Right);
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = scanWindow<Type>();
			constexpr size_t advance = width * window;
			constexpr uint64_t fullMask = width == 64 ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);
			const r zero = SetZero<Type, uint8_t>::invoke();

			const auto diff = [l, rp](size_t o) {
				return BitXor<Type>::invoke(Loadu<Type, uint8_t>::invoke(l + o), Loadu<Type, uint8_t>::invoke(rp + o));
			};
			const auto missBits = [zero](r v_Diff) { return (~byteEqualMask<Type>(v_Diff, zero)) & fullMask; };
			const auto sign = [l, rp](size_t o) { return l[o] < rp[o] ? -1 : 1; };

			size_t v = 0;
			for (; v + advance <= v_Size; v += advance) {
				r lanes[window];
				Unrolled::staticFor<0, window>([&](auto idx) { lanes[idx] = diff(v + idx * width); });
				r acc = lanes[0];
				Unrolled::staticFor<1, window>([&](auto idx) { acc = BitOr<Type>::invoke(acc, lanes[idx]); });
				if (TestZero<Type>::invoke(acc)) continue;
				for (size_t k = 0; k < window; ++k)
					if (const uint64_t m = missBits(lanes[k]))
						return sign(v + k * width + std::countr_zero(m));
			}
			for (; v + width <= v_Size; v += width)
				if (const uint64_t m = missBits(diff(v)))
					return sign(v + std::countr_zero(m));
			if (v != v_Size)
				if (const uint64_t m = missBits(diff(v_Size - width)))
					return sign(v_Size - width + std::countr_zero(m));
			return 0;
		}

		template<VType Type>
		const void* memFindByte(const void* STL_RESTRICT p_Ptr, size_t v_Size, uint8_t v_Byte) {
			using r = VIntrospect<Type>::template RType<uint8_t>;
			const auto* p = static_cast<const uint8_t*>(p_Ptr);
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = scanWindow<Type>();
			constexpr size_t advance = width * window;
			constexpr uint64_t fullMask = width == 64 ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);
			const r needle = Set1<Type, uint8_t>::invoke(v_Byte);

			const auto hitBits = [p, needle](size_t o) {
				return byteEqualMask<Type>(Loadu<Type, uint8_t>::invoke(p + o), needle) & fullMask;
			};

			size_t v = 0;
#if STL_AVX512_SUPPORT
			if constexpr (Type == VType::V_AVX512) {
				// Compares land in k-registers already, so OR the masks directly.
				for (; v + advance <= v_Size; v += advance) {
					uint64_t lanes[window];
					uint64_t any = 0;
					Unrolled::staticFor<0, window>([&](auto idx) { any |= lanes[idx] = hitBits(v + idx * width); });
					if (!any) continue;
					for (size_t k = 0; k < window; ++k)
						if (lanes[k]) return p + v + k * width + std::countr_zero(lanes[k]);
				}
			} else
#endif
			{
				for (; v + advance <= v_Size; v += advance) {
					r lanes[window];
					Unrolled::staticFor<0, window>([&](auto idx) {
						lanes[idx] = CompareEqual<Type, uint8_t>::invoke(Loadu<Type, uint8_t>::invoke(p + v + idx * width), needle);
					});
					r acc = lanes[0];
					Unrolled::staticFor<1, window>([&](auto idx) { acc = BitOr<Type>::invoke(acc, lanes[idx]); });
					if (TestZero<Type>::invoke(acc)) continue;
					for (size_t k = 0; k < window; ++k)
						if (const uint64_t m = MoveMask<Type, uint8_t>::invoke(lanes[k]))
							return p + v + k * width + std::countr_zero(m);
				}
			}
			for (; v + width <= v_Size; v += width)
				if (const uint64_t m = hitBits(v)) return p + v + std::countr_zero(m);
			if (v != v_Size)
				if (const uint64_t m = hitBits(v_Size - width)) return p + v_Size - width + std::countr_zero(m);
			return nullptr;
		}

		// Remainder is one overlapped window ending at v_Size; idempotent since every store writes v_Reg.
		// Requires v_Size >= one window.
		template<VType Type>
		void STL_FORCEINLINE storeWindowLoop(uint8_t* STL_RESTRICT p_Curr, size_t v_Size, const typename VIntrospect<Type>::template RType<uint8_t>& v_Reg) {
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = VIntrospect<Type>::UType::value;
			constexpr size_t advance = width * window;

			size_t v = 0;
			for (; v + advance <= v_Size; v += advance)
				Unrolled::staticFor<0, window>([p_Curr, v, v_Reg](auto idx) { Store<Type, uint8_t>::invoke(p_Curr + v + idx * width, v_Reg); });

			if (v != v_Size) {
				STL_ASSERT(v_Size >= advance);
				Unrolled::staticFor<0, window>([p_Curr, v_Size, v_Reg](auto idx) { Store<Type, uint8_t>::invoke(p_Curr + v_Size - advance + idx * width, v_Reg); });
			}
		}

		template<VType Type>
		void STL_FORCEINLINE streamWindowLoop(uint8_t* STL_RESTRICT p_Curr, size_t v_Size, const typename VIntrospect<Type>::template RType<uint8_t>& v_Reg) {
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = VIntrospect<Type>::UType::value;
			constexpr size_t advance = width * window;

			size_t v = 0;
			for (; v + advance <= v_Size; v += advance)
				Unrolled::staticFor<0, window>([p_Curr, v, v_Reg](auto idx) { Stream<Type, uint8_t>::invoke(p_Curr + v + idx * width, v_Reg); });

			if (v != v_Size) {
				STL_ASSERT(v_Size >= advance);
				Unrolled::staticFor<0, window>([p_Curr, v_Size, v_Reg](auto idx) { Stream<Type, uint8_t>::invoke(p_Curr + v_Size - advance + idx * width, v_Reg); });
			}
		}

#if STL_SSE_SUPPORT
		void STL_FORCEINLINE setMemoryZeroSSE(void* STL_RESTRICT p_Ptr, size_t v_Size) {
			using r = VIntrospect<VType::V_SSE>::RType<uint8_t>;
			const r m1 = SetZero<VType::V_SSE, uint8_t>::invoke();
			auto curr = static_cast<unsigned char*>(p_Ptr);
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([curr, m1](auto idx) { Store<VType::V_SSE, uint8_t>::invoke(curr + idx * 16, m1); });
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
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([curr, m1](auto idx) { Store<VType::V_AVX, uint8_t>::invoke(curr + idx * 32, m1); });
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
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([curr, m1](auto idx) { Store<VType::V_AVX512, uint8_t>::invoke(curr + idx * 64, m1); });
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
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Store<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16,
						Load<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16));
				});
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryUnalignedSSE(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_SSE_UNROLL_WINDOW * VIntrospect<VType::V_SSE>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Storeu<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16,
						Loadu<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16));
				});
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
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Store<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16,
						Load<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16));
				});
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
				Unrolled::staticFor<0, STL_SSE_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Stream<VType::V_SSE, uint8_t>::invoke(currDst + idx * 16,
						Load<VType::V_SSE, uint8_t>::invoke(currSrc + idx * 16));
				});
			}
		}
#endif

#if STL_AVX_SUPPORT
		// Capture by value so non-inlined instantiations read the pointers directly.
		void STL_FORCEINLINE copyMemoryAVX(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Store<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32,
						Load<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32));
				});
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryUnalignedAVX(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX_UNROLL_WINDOW * VIntrospect<VType::V_AVX>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Storeu<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32,
						Loadu<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32));
				});
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
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Store<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32,
						Load<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32));
				});
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
				Unrolled::staticFor<0, STL_AVX_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Stream<VType::V_AVX, uint8_t>::invoke(currDst + idx * 32,
						Load<VType::V_AVX, uint8_t>::invoke(currSrc + idx * 32));
				});
			}
		}
#endif

#if STL_AVX512_SUPPORT
		void STL_FORCEINLINE copyMemoryAVX512(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Store<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64,
						Load<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64));
				});
				currSrc += advance;
				currDst += advance;
			}
		}

		void STL_FORCEINLINE copyMemoryUnalignedAVX512(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			constexpr auto advance = STL_AVX512_UNROLL_WINDOW * VIntrospect<VType::V_AVX512>::kWidth;

			for (size_t v = 0; v < v_Size; v += advance) {
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Storeu<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64,
						Loadu<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64));
				});
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
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Store<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64,
						Load<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64));
				});
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
				Unrolled::staticFor<0, STL_AVX512_UNROLL_WINDOW>([currSrc, currDst](auto idx) {
					Stream<VType::V_AVX512, uint8_t>::invoke(currDst + idx * 64,
						Load<VType::V_AVX512, uint8_t>::invoke(currSrc + idx * 64));
				});
			}
		}
#endif

		// Canonical Ops with assumptions.

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

		// Forward streaming copy: aligned non-temporal stores to dst. v_SrcAligned picks an
		// aligned or unaligned load for src, since only dst can be aligned when offsets differ.
		template<VType Type, bool v_SrcAligned>
		void copyMemoryStream(const void* STL_RESTRICT p_Src, void* STL_RESTRICT p_Dst, size_t v_Size) {
			using SrcLoad = std::conditional_t<v_SrcAligned, Load<Type, uint8_t>, Loadu<Type, uint8_t>>;
			constexpr size_t width = VIntrospect<Type>::kWidth;
			constexpr size_t window = VIntrospect<Type>::UType::value;
			constexpr size_t advance = width * window;
			constexpr size_t prefetchAhead = 512;
			STL_ASSERT(v_Size % advance == 0);

			auto currSrc = static_cast<const unsigned char*>(p_Src);
			auto currDst = static_cast<unsigned char*>(p_Dst);
			for (size_t v = 0; v < v_Size; v += advance) {
				for (size_t p = 0; p < advance; p += 64)
					PrefetchRead::invoke<0>(currSrc + prefetchAhead + p);
				Unrolled::staticFor<0, window>([currSrc, currDst](auto idx) {
					Stream<Type, uint8_t>::invoke(currDst + idx * width, SrcLoad::invoke(currSrc + idx * width));
				});
				currSrc += advance;
				currDst += advance;
			}
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
