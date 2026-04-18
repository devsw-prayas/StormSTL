#pragma once
#include "StormSTL.h"
#include "VectorizationTraits.h"
#include "Stripe.h"

namespace StormSTL::Vectorization::msk256 {
	constexpr uint8_t  EnabledLane1B = 0xFF;
	constexpr uint8_t  DisabledLane1B = 0x00;

	constexpr uint16_t EnabledLane2B = 0xFFFF;
	constexpr uint16_t DisabledLane2B = 0x0000;

	constexpr uint32_t EnabledLane4B = 0xFFFFFFFFu;
	constexpr uint32_t DisabledLane4B = 0x00000000u;

	constexpr uint64_t EnabledLane8B = 0xFFFFFFFFFFFFFFFFull;
	constexpr uint64_t DisabledLane8B = 0x0000000000000000ull;

	using Mask256 = __m256i;

	struct STL_RUNTIME_API alignas(32) StripeMask final {
		Mask256 m_VectorMask;

		StripeMask() noexcept : m_VectorMask(_mm256_setzero_si256()) {}
		explicit StripeMask(Mask256 v_Raw) noexcept : m_VectorMask(v_Raw) {}

		StripeMask(const StripeMask&) = default;
		StripeMask& operator=(const StripeMask&) = default;

		StripeMask(StripeMask&&) noexcept = default;
		StripeMask& operator=(StripeMask&&) noexcept = default;
	};

	struct STL_RUNTIME_API alignas(32) StripeMask64 final {
		Mask256	m_VectorMask;
		StripeMask64() noexcept : m_VectorMask(_mm256_setzero_si256()) {}

		StripeMask64(const StripeMask64&) = default;
		StripeMask64& operator=(const StripeMask64&) = default;

		StripeMask64(StripeMask64&&) noexcept = default;
		StripeMask64& operator=(StripeMask64&&) noexcept = default;
	};

	struct STL_RUNTIME_API alignas(32) StripeMask32 final {
		Mask256	m_VectorMask;

		StripeMask32() noexcept : m_VectorMask(_mm256_setzero_si256()) {}
		StripeMask32(const StripeMask32&) = default;
		StripeMask32& operator=(const StripeMask32&) = default;

		StripeMask32(StripeMask32&&) noexcept = default;
		StripeMask32& operator=(StripeMask32&&) noexcept = default;
	};

	struct STL_RUNTIME_API alignas(32) StripeMask16 final {
		Mask256	m_VectorMask;

		StripeMask16() noexcept : m_VectorMask(_mm256_setzero_si256()) {}
		StripeMask16(const StripeMask16&) = default;
		StripeMask16& operator=(const StripeMask16&) = default;

		StripeMask16(StripeMask16&&) noexcept = default;
		StripeMask16& operator=(StripeMask16&&) noexcept = default;
	};

	struct STL_RUNTIME_API alignas(32) StripeMask8 final {
		Mask256	m_VectorMask;

		StripeMask8() noexcept : m_VectorMask(_mm256_setzero_si256()) {}
		StripeMask8(const StripeMask8&) = default;
		StripeMask8& operator=(const StripeMask8&) = default;

		StripeMask8(StripeMask8&&) noexcept = default;
		StripeMask8& operator=(StripeMask8&&) noexcept = default;
	};

	static_assert(sizeof(StripeMask) == 32,   "StripeMask must occupy exactly one AVX2 register");
	static_assert(alignof(StripeMask) == 32,  "StripeMask must be 32-byte aligned");
	static_assert(std::is_trivially_copyable_v<StripeMask>,  "StripeMask must be trivially copyable");
	static_assert(std::is_standard_layout_v<StripeMask>,     "StripeMask has invalid memory layout");

	static_assert(sizeof(StripeMask64) == 32, "StripeMask64 must occupy exactly one AVX2 register");
	static_assert(sizeof(StripeMask32) == 32, "StripeMask32 must occupy exactly one AVX2 register");
	static_assert(sizeof(StripeMask16) == 32, "StripeMask16 must occupy exactly one AVX2 register");
	static_assert(sizeof(StripeMask8) == 32, "StripeMask8 must occupy exactly one AVX2 register");

	static_assert(alignof(StripeMask64) == 32, "StripeMask64 must be 32-byte aligned");
	static_assert(alignof(StripeMask32) == 32, "StripeMask32 must be 32-byte aligned");
	static_assert(alignof(StripeMask16) == 32, "StripeMask16 must be 32-byte aligned");
	static_assert(alignof(StripeMask8) == 32, "StripeMask8 must be 32-byte aligned");

	static_assert(std::is_trivially_copyable_v<StripeMask64>, "StripeMask64 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<StripeMask32>, "StripeMask32 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<StripeMask16>, "StripeMask16 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<StripeMask8>, "StripeMask8 must be trivially copyable");

	static_assert(std::is_standard_layout_v<StripeMask64>, "StripeMask64 has invalid memory layout");
	static_assert(std::is_standard_layout_v<StripeMask32>, "StripeMask32 has invalid memory layout");
	static_assert(std::is_standard_layout_v<StripeMask16>, "StripeMask16 has invalid memory layout");
	static_assert(std::is_standard_layout_v<StripeMask8>, "StripeMask8 has invalid memory layout");
}

namespace StormSTL::Vectorization::Traits {
	template<>
	struct VectorizationIntrospect<msk256::StripeMask> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 0;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit Generic Mask (untyped)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct STL_RUNTIME_API IsVectorMask<msk256::StripeMask> final : std::true_type {};

	template<>
	struct VectorizationIntrospect<msk256::StripeMask64> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 4;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit 4 Lane Mask (64 bit per lane)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe64> {
		using type = msk256::StripeMask64;
	};

	template<>
	struct VectorizeMaskType<v256::StripeU64> {
		using type = msk256::StripeMask64;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe64f> {
		using type = msk256::StripeMask64;
	};

	template<>
	struct VectorizationIntrospect<msk256::StripeMask32> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 8;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit 8 Lane Mask (32 bit per lane)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe32> {
		using type = msk256::StripeMask32;
	};

	template<>
	struct VectorizeMaskType<v256::StripeU32> {
		using type = msk256::StripeMask32;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe32f> {
		using type = msk256::StripeMask32;
	};

	template<>
	struct VectorizationIntrospect<msk256::StripeMask16> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 16;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit 16 Lane Mask (16 bit per lane)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe16> {
		using type = msk256::StripeMask16;
	};

	template<>
	struct VectorizeMaskType<v256::StripeU16> {
		using type = msk256::StripeMask16;
	};

	template<>
	struct VectorizationIntrospect<msk256::StripeMask8> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 32;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit 32 Lane Mask (8 bit per lane)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe8> {
		using type = msk256::StripeMask8;
	};

	template<>
	struct VectorizeMaskType<v256::StripeU8> {
		using type = msk256::StripeMask8;
	};

	template<>
	struct STL_RUNTIME_API IsVectorMask<msk256::StripeMask64> final : std::true_type {};

	template<>
	struct STL_RUNTIME_API IsVectorMask<msk256::StripeMask32> final : std::true_type {};	
	
	template<>
	struct STL_RUNTIME_API IsVectorMask<msk256::StripeMask16> final : std::true_type {};

	template<>
	struct STL_RUNTIME_API IsVectorMask<msk256::StripeMask8> final : std::true_type {};

}

namespace StormSTL::Vectorization::msk256 {
	static_assert(256 / Traits::VectorizationIntrospect<StripeMask64>::s_Lanes == 64,
		"StripeMask64 lane-width invariant violated (expected 64 bits per lane)");
	static_assert(256 / Traits::VectorizationIntrospect<StripeMask32>::s_Lanes == 32,
		"StripeMask32 lane-width invariant violated (expected 32 bits per lane)");
	static_assert(256 / Traits::VectorizationIntrospect<StripeMask16>::s_Lanes == 16, 
		"StripeMask16 lane-width invariant violated (expected 16 bits per lane)");
	static_assert(256 / Traits::VectorizationIntrospect<StripeMask8>::s_Lanes == 8, 
		"StripeMask8 lane-width invariant violated (expected 8 bits per lane)");
}