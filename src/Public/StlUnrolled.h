#pragma once
#include "StormSTL.h"

namespace Stl::Unrolled {
	template<size_t Idx>
	using index = std::integral_constant<size_t, Idx>;

	template<size_t Begin, size_t End, typename Fn>
	constexpr void staticFor(Fn&& u_Func) {
		[&]<size_t... I>(std::index_sequence<I...>) { (u_Func(index<Begin + I>{}), ...); }(std::make_index_sequence<End - Begin>{});
	}

}