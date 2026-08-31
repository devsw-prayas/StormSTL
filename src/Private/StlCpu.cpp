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
			CacheSizes out{ 0, 0 };
#if defined(_WIN32)
			DWORD len = 0;
			GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);

			alignas(alignof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)) unsigned char buf[8192];
			if (len == 0 || len > sizeof(buf)) return out;
			if (!GetLogicalProcessorInformationEx(RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf), &len)) return out;

			unsigned llcLevel = 0;
			for (DWORD off = 0; off < len;) {
				const auto* rec = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf + off);
				if (rec->Relationship == RelationCache) {
					const CACHE_RELATIONSHIP& cache = rec->Cache;
					if ((cache.Type == CacheUnified || cache.Type == CacheData) && cache.CacheSize != 0) {
						if (cache.Level == 1 && (out.m_L1D == 0 || cache.CacheSize < out.m_L1D)) out.m_L1D = cache.CacheSize;
						if (cache.Level > llcLevel || (cache.Level == llcLevel && cache.CacheSize > out.m_Llc)) {
							llcLevel = cache.Level;
							out.m_Llc = cache.CacheSize;
						}
					}
				}
				off += rec->Size;
			}
#elif defined(__linux__)
			long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
			if (l1 > 0) out.m_L1D = static_cast<size_t>(l1);
			long llc = sysconf(_SC_LEVEL3_CACHE_SIZE);
			if (llc <= 0) llc = sysconf(_SC_LEVEL2_CACHE_SIZE);
			if (llc > 0) out.m_Llc = static_cast<size_t>(llc);
#endif
			return out;
		}
	}

	size_t detectNonTemporalThreshold() noexcept {
		const size_t llc = probeCaches().m_Llc;
		return llc != 0 ? llc / 2 : static_cast<size_t>(STL_NT_JUMP_SIZE);
	}

	size_t detectStosThreshold() noexcept {
		const size_t l1d = probeCaches().m_L1D;
		return l1d != 0 ? l1d : static_cast<size_t>(32) * 1024;
	}
}