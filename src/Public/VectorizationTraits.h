#pragma once
#include "StormSTL.h"

namespace StormSTL::Vectorization::Traits {

	enum class STL_RUNTIME_API VectorizationBackend : uint8_t {
		UNKNOWN,
		AVX2
	};

	template<VectorizationBackend Backend = VectorizationBackend::UNKNOWN>
	struct STL_RUNTIME_API ImplementationType final {
		static constexpr size_t s_Width = 0;
		static constexpr size_t s_Alignment = 0;
	};

	template<VectorizationBackend Backend>
	inline constexpr STL_RUNTIME_API size_t VectorizationWidth = ImplementationType<Backend>::s_Width;

	template<VectorizationBackend Backend>
	inline constexpr STL_RUNTIME_API size_t VectorizationAlignment = ImplementationType<Backend>::s_Alignment;

	template<>
	struct STL_RUNTIME_API ImplementationType<VectorizationBackend::AVX2> final{
		static constexpr size_t s_Width = 256;
		static constexpr size_t s_Alignment = 32;
	};

	template<typename T, typename = void>
	struct STL_RUNTIME_API VectorizeType final {
		using type = void;
	};

	template<typename T>
	using VectorizedForm = VectorizeType<T>::type;

	template<typename T, typename = void>
	struct STL_RUNTIME_API VectorizeMaskType final {
		using type = void;
	};

	template<typename T>
	using  VectorizedMask = VectorizeMaskType<T>::type;

	template<typename T, typename = void>
	struct STL_RUNTIME_API ScalarType final {
		using type = void;
	};

	template<typename T>
	using ScalarForm = ScalarType<T>::type;

	struct Blank final {};

	template<typename T, typename = void>
	struct STL_RUNTIME_API IsVectorStripe final : std::false_type {};

	template<typename T, typename = void>
	struct STL_RUNTIME_API IsVectorMask final : std::false_type {};

	template<typename T>
	inline constexpr STL_RUNTIME_API bool IsVectorStripeV = IsVectorStripe<T>::value;

	template<typename T>
	inline constexpr STL_RUNTIME_API bool IsVectorMaskV = IsVectorMask<T>::value;


	template<typename T = Blank>
	struct STL_RUNTIME_API VectorizationIntrospect final {
		static constexpr size_t s_Alignment = 0;
		static constexpr size_t s_Lanes = 0;
		static constexpr size_t s_Width = 0;
		static constexpr const char* s_Name = "Invalid Type";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::UNKNOWN;
	};

	template<typename T = Blank>
	inline constexpr STL_RUNTIME_API size_t IntrospectAlignment = VectorizationIntrospect<T>::s_Alignment;

	template<typename T = Blank>
	inline constexpr STL_RUNTIME_API size_t IntrospectLanes = VectorizationIntrospect<T>::s_Lanes;

	template<typename T = Blank>
	inline constexpr STL_RUNTIME_API size_t IntrospectWidth = VectorizationIntrospect<T>::s_Width;

	template<typename T = Blank>
	inline constexpr STL_RUNTIME_API const char* IntrospectTypeName = VectorizationIntrospect<T>::s_Name;

	template<typename T = Blank>
	inline constexpr STL_RUNTIME_API VectorizationBackend IntrospectBackend = VectorizationIntrospect<T>::s_Backend;

	template<typename T>
	inline constexpr STL_RUNTIME_API bool TemplateFalseV = false;
}
