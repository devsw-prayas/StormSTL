#include "StlCpu.h"

#include "StormSTL.h"

#define ALLOW_SYSCALL
#include "StlSyscalls.h"

namespace Stl::Internal {
	namespace {
		struct CacheSizes {
			size_t m_L1D;
			size_t m_Llc;
		};

		CacheSizes probeCaches() noexcept {
			CacheSizes l_Out{ 0, 0 };
#if defined(_WIN32)
			DWORD v_Len = 0;
			GetLogicalProcessorInformationEx(RelationCache, nullptr, &v_Len);

			alignas(alignof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)) unsigned char l_Buf[8192];
			if (v_Len == 0 || v_Len > sizeof(l_Buf)) return l_Out;
			if (!GetLogicalProcessorInformationEx(RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(l_Buf), &v_Len)) return l_Out;

			unsigned l_LlcLevel = 0;
			for (DWORD l_Off = 0; l_Off < v_Len;) {
				const auto* p_Rec = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(l_Buf + l_Off);
				if (p_Rec->Relationship == RelationCache) {
					const CACHE_RELATIONSHIP& r_Cache = p_Rec->Cache;
					if ((r_Cache.Type == CacheUnified || r_Cache.Type == CacheData) && r_Cache.CacheSize != 0) {
						if (r_Cache.Level == 1 && (l_Out.m_L1D == 0 || r_Cache.CacheSize < l_Out.m_L1D)) l_Out.m_L1D = r_Cache.CacheSize;
						if (r_Cache.Level > l_LlcLevel || (r_Cache.Level == l_LlcLevel && r_Cache.CacheSize > l_Out.m_Llc)) {
							l_LlcLevel = r_Cache.Level;
							l_Out.m_Llc = r_Cache.CacheSize;
						}
					}
				}
				l_Off += p_Rec->Size;
			}
#elif defined(__linux__)
			long l_L1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
			if (l_L1 > 0) l_Out.m_L1D = static_cast<size_t>(l_L1);
			long l_Llc = sysconf(_SC_LEVEL3_CACHE_SIZE);
			if (l_Llc <= 0) l_Llc = sysconf(_SC_LEVEL2_CACHE_SIZE);
			if (l_Llc > 0) l_Out.m_Llc = static_cast<size_t>(l_Llc);
#endif
			return l_Out;
		}
	}

	size_t detectNonTemporalThreshold() noexcept {
		const size_t l_Llc = probeCaches().m_Llc;
		return l_Llc != 0 ? l_Llc / 2 : static_cast<size_t>(STL_NT_JUMP_SIZE);
	}

	size_t detectStosThreshold() noexcept {
		const size_t l_L1D = probeCaches().m_L1D;
		return l_L1D != 0 ? l_L1D : static_cast<size_t>(32) * 1024;
	}
}