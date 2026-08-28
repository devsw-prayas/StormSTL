#pragma once

#include <atomic>
#include <cstddef>

namespace Stl::Internal {
	size_t detectNonTemporalThreshold() noexcept;
	size_t detectStosThreshold() noexcept;

	inline std::atomic<size_t> g_NonTemporalThreshold{ 0 };
	inline std::atomic<size_t> g_StosThreshold{ 0 };

	inline size_t nonTemporalThreshold() noexcept {
		size_t v_Value = g_NonTemporalThreshold.load(std::memory_order_relaxed);
		if (v_Value == 0) {
			v_Value = detectNonTemporalThreshold();
			g_NonTemporalThreshold.store(v_Value, std::memory_order_relaxed);
		}
		return v_Value;
	}

	inline size_t stosThreshold() noexcept {
		size_t v_Value = g_StosThreshold.load(std::memory_order_relaxed);
		if (v_Value == 0) {
			v_Value = detectStosThreshold();
			g_StosThreshold.store(v_Value, std::memory_order_relaxed);
		}
		return v_Value;
	}
}
