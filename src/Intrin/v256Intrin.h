/*
* Copyright (c) 2025 StormWeaver
*
* This file is part of the Standard Vector Backend
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
#define ALLOW_VECTOR_INTRIN_128
#include <v128Intrin.h>

#ifndef ALLOW_VECTOR_INTRIN_256
#error "Explicitly define ALLOW_VECTOR_INTRIN_256 before including this header."
#else
#if !defined(__AVX2__) && !defined(_vectorbackend_EDITOR_MODE)
#error "_vectorbackend SIMD Vector Backend requires AVX2 flag to be enabled during compilation"
#else
namespace StormSTL::Vectorization::Intrinsic::v256 {
	using namespace v128;
	using r256i = __m256i;
	using r256fp32 = __m256;
	using r256fp64 = __m256d;

	// AVX: Vector construction and constants
	// float32 (8 lanes) and float64 (4 lanes)
	// zero, broadcast, and explicit lane construction
	//
	// AVX2: Vector construction (integer / __m256i)
	// broadcast, explicit lane construction, and ordered variants
	//
	// AVX: Vector load and store (float32 / float64)
	// aligned and unaligned memory access
	//
	// AVX2: Vector load and store (integer / __m256i)
	//
	// AVX / AVX2: Masked load and store
	// partial memory access controlled by per-lane masks
	//
	// AVX: Basic arithmetic operations (float32 / float64)
	// add, subtract, multiply, divide
	//
	// AVX2: Basic arithmetic operations (integer)
	// wraparound and saturating add/sub, integer multiply variants
	//
	// FMA (FMA3): 256-bit fused multiply-add operations
	// single-rounding multiply-add/subtract variants
	//
	// AVX2: Fused integer arithmetic
	// pairwise multiply-add operations
	//
	// AVX: Min/max operations (float32 / float64)
	//
	// AVX2: Min/max operations (integer)
	// signed and unsigned variants, width-explicit
	//
	// AVX: Vector comparisons (float32 / float64)
	// immediate-controlled predicate generation, full-lane masks
	//
	// AVX2: Vector comparisons (integer)
	// equality and greater-than, signed semantics
	//
	// AVX / AVX2: Predicate extraction
	// movemask operations
	//
	// AVX: Test operations
	// zero / containment / mixed-bit tests for float, double, and integer
	//
	// AVX: Bitwise operations (float32 / float64)
	//
	// AVX2: Bitwise operations (integer)
	//
	// AVX2: Vector shifts (integer)
	// immediate shifts and variable shifts
	// logical and arithmetic variants
	//
	// AVX: Vector shuffle operations (float32 / float64)
	//
	// AVX2: Vector shuffle operations (integer)
	//
	// AVX: Unpack operations
	// interleave low/high lanes for integer and floating-point vectors
	//
	// AVX / AVX2: Blend operations
	// immediate and mask-controlled lane selection
	//
	// AVX / AVX2: Permute operations
	// variable-index and immediate-controlled permutations
	//
	// AVX / AVX2: 128-bit lane permutation operations
	// cross-lane rearrangement within 256-bit vectors
	//
	// AVX / AVX2: Insert and extract 128-bit lanes
	//
	// AVX2: Alignment operations
	//
	// AVX: Horizontal operations (float32 / float64)
	//
	// AVX2: Horizontal operations (integer)
	//
	// AVX2: Sign and absolute-value operations
	//
	// AVX: Square root, reciprocal, and reciprocal square root
	//
	// AVX: Rounding operations
	//
	// AVX2: Packing operations
	//
	// AVX2: Broadcast operations
	//
	// AVX / AVX2: Type conversion operations
	//
	// AVX: Cast operations (zero-cost reinterpretation)
	//
	// AVX / AVX2: Gather operations
	// indexed memory reads with optional masking
	//
	// AVX: Streaming stores and utility operations

	// zero

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp32 zeroSinglePrecision() {
		return _mm256_setzero_ps();
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp64 zeroDoublePrecision() {
		return _mm256_setzero_pd();
	}

	// broadcast

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp32 setOne(float v) {
		return _mm256_set1_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp64 setOne(double v) {
		return _mm256_set1_pd(v);
	}

	// reversed order (INTERNAL / DANGEROUS)

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp32 setReversed(
			float v7, float v6, float v5, float v4,
			float v3, float v2, float v1, float v0
		) {
		return _mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp64 setReversed(
			double v3, double v2, double v1, double v0
		) {
		return _mm256_set_pd(v3, v2, v1, v0);
	}

	// regular order (PUBLIC / SANE)

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp32 set(
			float v0, float v1, float v2, float v3,
			float v4, float v5, float v6, float v7
		) {
		return _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp64 set(
			double v0, double v1, double v2, double v3
		) {
		return _mm256_setr_pd(v0, v1, v2, v3);
	}


// zero

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i zero() {
		return _mm256_setzero_si256();
	}

	// broadcast

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setOne(int8_t v) {
		return _mm256_set1_epi8(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setOne(int16_t v) {
		return _mm256_set1_epi16(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setOne(int32_t v) {
		return _mm256_set1_epi32(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setOne(int64_t v) {
		return _mm256_set1_epi64x(v);
	}

	// reversed order (INTERNAL / DANGEROUS)

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setReversed(
			int8_t v31, int8_t v30, int8_t v29, int8_t v28,
			int8_t v27, int8_t v26, int8_t v25, int8_t v24,
			int8_t v23, int8_t v22, int8_t v21, int8_t v20,
			int8_t v19, int8_t v18, int8_t v17, int8_t v16,
			int8_t v15, int8_t v14, int8_t v13, int8_t v12,
			int8_t v11, int8_t v10, int8_t v9, int8_t v8,
			int8_t v7, int8_t v6, int8_t v5, int8_t v4,
			int8_t v3, int8_t v2, int8_t v1, int8_t v0
		) {
		return _mm256_set_epi8(
			v31, v30, v29, v28, v27, v26, v25, v24,
			v23, v22, v21, v20, v19, v18, v17, v16,
			v15, v14, v13, v12, v11, v10, v9, v8,
			v7, v6, v5, v4, v3, v2, v1, v0
		);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setReversed(
			int16_t v15, int16_t v14, int16_t v13, int16_t v12,
			int16_t v11, int16_t v10, int16_t v9, int16_t v8,
			int16_t v7, int16_t v6, int16_t v5, int16_t v4,
			int16_t v3, int16_t v2, int16_t v1, int16_t v0
		) {
		return _mm256_set_epi16(
			v15, v14, v13, v12, v11, v10, v9, v8,
			v7, v6, v5, v4, v3, v2, v1, v0
		);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setReversed(
			int32_t v7, int32_t v6, int32_t v5, int32_t v4,
			int32_t v3, int32_t v2, int32_t v1, int32_t v0
		) {
		return _mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setReversed(
			int64_t v3, int64_t v2, int64_t v1, int64_t v0
		) {
		return _mm256_set_epi64x(v3, v2, v1, v0);
	}

	// regular order (PUBLIC / SANE)

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i set(
			int8_t  v0, int8_t  v1, int8_t  v2, int8_t  v3,
			int8_t  v4, int8_t  v5, int8_t  v6, int8_t  v7,
			int8_t  v8, int8_t  v9, int8_t  v10, int8_t  v11,
			int8_t  v12, int8_t  v13, int8_t  v14, int8_t  v15,
			int8_t  v16, int8_t  v17, int8_t  v18, int8_t  v19,
			int8_t  v20, int8_t  v21, int8_t  v22, int8_t  v23,
			int8_t  v24, int8_t  v25, int8_t  v26, int8_t  v27,
			int8_t  v28, int8_t  v29, int8_t  v30, int8_t  v31
		) {
		return _mm256_setr_epi8(
			v0, v1, v2, v3, v4, v5, v6, v7,
			v8, v9, v10, v11, v12, v13, v14, v15,
			v16, v17, v18, v19, v20, v21, v22, v23,
			v24, v25, v26, v27, v28, v29, v30, v31
		);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i set(
			int16_t v0, int16_t v1, int16_t v2, int16_t v3,
			int16_t v4, int16_t v5, int16_t v6, int16_t v7,
			int16_t v8, int16_t v9, int16_t v10, int16_t v11,
			int16_t v12, int16_t v13, int16_t v14, int16_t v15
		) {
		return _mm256_setr_epi16(
			v0, v1, v2, v3, v4, v5, v6, v7,
			v8, v9, v10, v11, v12, v13, v14, v15
		);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i set(
			int32_t v0, int32_t v1, int32_t v2, int32_t v3,
			int32_t v4, int32_t v5, int32_t v6, int32_t v7
		) {
		return _mm256_setr_epi32(v0, v1, v2, v3, v4, v5, v6, v7);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i set(
			int64_t v0, int64_t v1, int64_t v2, int64_t v3
		) {
		return _mm256_setr_epi64x(v0, v1, v2, v3);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r256fp32 loadAligned(const float* p) {
		return _mm256_load_ps(p);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r256fp32 loadUnaligned(const float* p) {
		return _mm256_loadu_ps(p);
	}

	STL_FORCEINLINE
		void storeAligned(float* p, r256fp32 v) {
		_mm256_store_ps(p, v);
	}

	STL_FORCEINLINE
		void storeUnaligned(float* p, r256fp32 v) {
		_mm256_storeu_ps(p, v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r256fp64 loadAligned(const double* p) {
		return _mm256_load_pd(p);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r256fp64 loadUnaligned(const double* p) {
		return _mm256_loadu_pd(p);
	}

	STL_FORCEINLINE
		void storeAligned(double* p, r256fp64 v) {
		_mm256_store_pd(p, v);
	}

	STL_FORCEINLINE
		void storeUnaligned(double* p, r256fp64 v) {
		_mm256_storeu_pd(p, v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r256i loadAligned(const r256i* p) {
		return _mm256_load_si256(p);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r256i loadUnaligned(const r256i* p) {
		return _mm256_loadu_si256(p);
	}

	STL_FORCEINLINE
		void storeAligned(r256i* p, r256i v) {
		_mm256_store_si256(p, v);
	}

	STL_FORCEINLINE
		void storeUnaligned(r256i* p, r256i v) {
		_mm256_storeu_si256(p, v);
	}

	// float32

	STL_FORCEINLINE
		r256fp32 maskLoad(const float* p, r256i mask) {
		return _mm256_maskload_ps(p, mask);
	}

	STL_FORCEINLINE
		void maskStore(float* p, r256i mask, r256fp32 v) {
		_mm256_maskstore_ps(p, mask, v);
	}

	// float64

	STL_FORCEINLINE
		r256fp64 maskLoad(const double* p, r256i mask) {
		return _mm256_maskload_pd(p, mask);
	}

	STL_FORCEINLINE
		void maskStore(double* p, r256i mask, r256fp64 v) {
		_mm256_maskstore_pd(p, mask, v);
	}

	// integer (32-bit lanes)

	STL_FORCEINLINE
		r256i maskLoad32(const int32_t* p, r256i mask) {
		return _mm256_maskload_epi32(p, mask);
	}

	STL_FORCEINLINE
		void maskStore32(int32_t* p, r256i mask, r256i v) {
		_mm256_maskstore_epi32(p, mask, v);
	}

	// integer (64-bit lanes)

	STL_FORCEINLINE
		r256i maskLoad64(const int64_t* p, r256i mask) {
		return _mm256_maskload_epi64(p, mask);
	}

	STL_FORCEINLINE
		void maskStore64(int64_t* p, r256i mask, r256i v) {
		_mm256_maskstore_epi64(p, mask, v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp32 add(r256fp32 a, r256fp32 b) {
		return _mm256_add_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp32 sub(r256fp32 a, r256fp32 b) {
		return _mm256_sub_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp32 mul(r256fp32 a, r256fp32 b) {
		return _mm256_mul_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp32 div(r256fp32 a, r256fp32 b) {
		return _mm256_div_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp64 add(r256fp64 a, r256fp64 b) {
		return _mm256_add_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp64 sub(r256fp64 a, r256fp64 b) {
		return _mm256_sub_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp64 mul(r256fp64 a, r256fp64 b) {
		return _mm256_mul_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp64 div(r256fp64 a, r256fp64 b) {
		return _mm256_div_pd(a, b);
	}

	// wraparound add

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i add8(r256i a, r256i b) { return _mm256_add_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i add16(r256i a, r256i b) { return _mm256_add_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i add32(r256i a, r256i b) { return _mm256_add_epi32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i add64(r256i a, r256i b) { return _mm256_add_epi64(a, b); }

	// wraparound sub

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i sub8(r256i a, r256i b) { return _mm256_sub_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i sub16(r256i a, r256i b) { return _mm256_sub_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i sub32(r256i a, r256i b) { return _mm256_sub_epi32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i sub64(r256i a, r256i b) { return _mm256_sub_epi64(a, b); }

	// saturating add

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i addSaturate8(r256i a, r256i b) { return _mm256_adds_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i addSaturate16(r256i a, r256i b) { return _mm256_adds_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i addSaturateU8(r256i a, r256i b) { return _mm256_adds_epu8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i addSaturateU16(r256i a, r256i b) { return _mm256_adds_epu16(a, b); }

	// saturating sub

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i subSaturate8(r256i a, r256i b) { return _mm256_subs_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i subSaturate16(r256i a, r256i b) { return _mm256_subs_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i subSaturateU8(r256i a, r256i b) { return _mm256_subs_epu8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i subSaturateU16(r256i a, r256i b) { return _mm256_subs_epu16(a, b); }

	// integer multiply

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i mulLow16(r256i a, r256i b) { return _mm256_mullo_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i mulLow32(r256i a, r256i b) { return _mm256_mullo_epi32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i mulHigh16(r256i a, r256i b) { return _mm256_mulhi_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i mulHighU16(r256i a, r256i b) { return _mm256_mulhi_epu16(a, b); }

	// widening multiplies (even lanes only)

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i mulUnsigned32To64(r256i a, r256i b) {
		return _mm256_mul_epu32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i mulSigned32To64(r256i a, r256i b) {
		return _mm256_mul_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp32 fmadd(r256fp32 a, r256fp32 b, r256fp32 c) {
		return _mm256_fmadd_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp64 fmadd(r256fp64 a, r256fp64 b, r256fp64 c) {
		return _mm256_fmadd_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp32 fmsub(r256fp32 a, r256fp32 b, r256fp32 c) {
		return _mm256_fmsub_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp64 fmsub(r256fp64 a, r256fp64 b, r256fp64 c) {
		return _mm256_fmsub_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp32 fnmadd(r256fp32 a, r256fp32 b, r256fp32 c) {
		return _mm256_fnmadd_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp64 fnmadd(r256fp64 a, r256fp64 b, r256fp64 c) {
		return _mm256_fnmadd_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp32 fnmsub(r256fp32 a, r256fp32 b, r256fp32 c) {
		return _mm256_fnmsub_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp64 fnmsub(r256fp64 a, r256fp64 b, r256fp64 c) {
		return _mm256_fnmsub_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp32 fmaddsub(r256fp32 a, r256fp32 b, r256fp32 c) {
		return _mm256_fmaddsub_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp64 fmaddsub(r256fp64 a, r256fp64 b, r256fp64 c) {
		return _mm256_fmaddsub_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp32 fmsubadd(r256fp32 a, r256fp32 b, r256fp32 c) {
		return _mm256_fmsubadd_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r256fp64 fmsubadd(r256fp64 a, r256fp64 b, r256fp64 c) {
		return _mm256_fmsubadd_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i mulAdd16(r256i a, r256i b) {
		return _mm256_madd_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i mulAddUnsignedByteSignedByte(r256i a, r256i b) {
		return _mm256_maddubs_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp32 min(r256fp32 a, r256fp32 b) {
		return _mm256_min_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp32 max(r256fp32 a, r256fp32 b) {
		return _mm256_max_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp64 min(r256fp64 a, r256fp64 b) {
		return _mm256_min_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256fp64 max(r256fp64 a, r256fp64 b) {
		return _mm256_max_pd(a, b);
	}

	// signed

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i min8(r256i a, r256i b) { return _mm256_min_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i min16(r256i a, r256i b) { return _mm256_min_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i min32(r256i a, r256i b) { return _mm256_min_epi32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i max8(r256i a, r256i b) { return _mm256_max_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i max16(r256i a, r256i b) { return _mm256_max_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i max32(r256i a, r256i b) { return _mm256_max_epi32(a, b); }

	// unsigned

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i minUnsigned8(r256i a, r256i b) { return _mm256_min_epu8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i minUnsigned16(r256i a, r256i b) { return _mm256_min_epu16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i minUnsigned32(r256i a, r256i b) { return _mm256_min_epu32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i maxUnsigned8(r256i a, r256i b) { return _mm256_max_epu8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i maxUnsigned16(r256i a, r256i b) { return _mm256_max_epu16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r256i maxUnsigned32(r256i a, r256i b) { return _mm256_max_epu32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256fp32 cmp(r256fp32 a, r256fp32 b, int imm) {
		return _mm256_cmp_ps(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256fp64 cmp(r256fp64 a, r256fp64 b, int imm) {
		return _mm256_cmp_pd(a, b, imm);
	}

	// equality

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256i cmpEqual8(r256i a, r256i b) { return _mm256_cmpeq_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256i cmpEqual16(r256i a, r256i b) { return _mm256_cmpeq_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256i cmpEqual32(r256i a, r256i b) { return _mm256_cmpeq_epi32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256i cmpEqual64(r256i a, r256i b) { return _mm256_cmpeq_epi64(a, b); }

	// greater-than (signed)

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256i cmpGreater8(r256i a, r256i b) { return _mm256_cmpgt_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256i cmpGreater16(r256i a, r256i b) { return _mm256_cmpgt_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256i cmpGreater32(r256i a, r256i b) { return _mm256_cmpgt_epi32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r256i cmpGreater64(r256i a, r256i b) { return _mm256_cmpgt_epi64(a, b); }

	// AVX2 SSE
	// legacy SSE compare (kept for uniform API surface)

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmp(r128f32 a, r128f32 b, int imm) {
		return _mm_cmp_ps(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmp(r128f64 a, r128f64 b, int imm) {
		return _mm_cmp_pd(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Predicate extraction results must not be discarded")
		int movemask(r256fp32 v) {
		return _mm256_movemask_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Predicate extraction results must not be discarded")
		int movemask(r256fp64 v) {
		return _mm256_movemask_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Predicate extraction results must not be discarded")
		int movemask(r256i v) {
		return _mm256_movemask_epi8(v);
	}

	// zero test

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testZero(r256fp32 a, r256fp32 b) {
		return _mm256_testz_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testZero(r256fp64 a, r256fp64 b) {
		return _mm256_testz_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testZero(r256i a, r256i b) {
		return _mm256_testz_si256(a, b);
	}

	// containment test

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testAllContained(r256fp32 a, r256fp32 b) {
		return _mm256_testc_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testAllContained(r256fp64 a, r256fp64 b) {
		return _mm256_testc_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testAllContained(r256i a, r256i b) {
		return _mm256_testc_si256(a, b);
	}

	// mixed ones / zeros test

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testMixed(r256fp32 a, r256fp32 b) {
		return _mm256_testnzc_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testMixed(r256fp64 a, r256fp64 b) {
		return _mm256_testnzc_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testMixed(r256i a, r256i b) {
		return _mm256_testnzc_si256(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256fp32 bitAnd(r256fp32 a, r256fp32 b) {
		return _mm256_and_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256fp64 bitAnd(r256fp64 a, r256fp64 b) {
		return _mm256_and_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256fp32 bitOr(r256fp32 a, r256fp32 b) {
		return _mm256_or_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256fp64 bitOr(r256fp64 a, r256fp64 b) {
		return _mm256_or_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256fp32 bitXor(r256fp32 a, r256fp32 b) {
		return _mm256_xor_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256fp64 bitXor(r256fp64 a, r256fp64 b) {
		return _mm256_xor_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256fp32 bitAndNot(r256fp32 a, r256fp32 b) {
		return _mm256_andnot_ps(a, b); // (~a) & b
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256fp64 bitAndNot(r256fp64 a, r256fp64 b) {
		return _mm256_andnot_pd(a, b); // (~a) & b
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256i bitAnd(r256i a, r256i b) {
		return _mm256_and_si256(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256i bitOr(r256i a, r256i b) {
		return _mm256_or_si256(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256i bitXor(r256i a, r256i b) {
		return _mm256_xor_si256(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r256i bitAndNot(r256i a, r256i b) {
		return _mm256_andnot_si256(a, b); // (~a) & b
	}

	// logical left

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftLeft16(r256i v, int imm) {
		return _mm256_slli_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftLeft32(r256i v, int imm) {
		return _mm256_slli_epi32(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftLeft64(r256i v, int imm) {
		return _mm256_slli_epi64(v, imm);
	}

	// logical right

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightLogical16(r256i v, int imm) {
		return _mm256_srli_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightLogical32(r256i v, int imm) {
		return _mm256_srli_epi32(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightLogical64(r256i v, int imm) {
		return _mm256_srli_epi64(v, imm);
	}

	// arithmetic right

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightArithmetic16(r256i v, int imm) {
		return _mm256_srai_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightArithmetic32(r256i v, int imm) {
		return _mm256_srai_epi32(v, imm);
	}

	// logical left

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftLeft16(r256i v, r128i count) {
		return _mm256_sll_epi16(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftLeft32(r256i v, r128i count) {
		return _mm256_sll_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftLeft64(r256i v, r128i count) {
		return _mm256_sll_epi64(v, count);
	}

	// logical right

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightLogical16(r256i v, r128i count) {
		return _mm256_srl_epi16(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightLogical32(r256i v, r128i count) {
		return _mm256_srl_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightLogical64(r256i v, r128i count) {
		return _mm256_srl_epi64(v, count);
	}

	// arithmetic right

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightArithmetic16(r256i v, r128i count) {
		return _mm256_sra_epi16(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightArithmetic32(r256i v, r128i count) {
		return _mm256_sra_epi32(v, count);
	}

	// 128-bit legacy forms

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftLeftVariable32(r128i v, r128i count) {
		return _mm_sllv_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftLeftVariable64(r128i v, r128i count) {
		return _mm_sllv_epi64(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightLogicalVariable32(r128i v, r128i count) {
		return _mm_srlv_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightLogicalVariable64(r128i v, r128i count) {
		return _mm_srlv_epi64(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightArithmeticVariable32(r128i v, r128i count) {
		return _mm_srav_epi32(v, count);
	}

	// 256-bit AVX2 forms

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftLeftVariable32(r256i v, r256i count) {
		return _mm256_sllv_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftLeftVariable64(r256i v, r256i count) {
		return _mm256_sllv_epi64(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightLogicalVariable32(r256i v, r256i count) {
		return _mm256_srlv_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightLogicalVariable64(r256i v, r256i count) {
		return _mm256_srlv_epi64(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightArithmeticVariable32(r256i v, r256i count) {
		return _mm256_srav_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r256i shiftRightArithmeticVariable64(r256i v, r256i count) {
		return _mm256_srav_epi64(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r256fp32 shuffle(r256fp32 a, r256fp32 b, int imm) {
		return _mm256_shuffle_ps(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r256fp64 shuffle(r256fp64 a, r256fp64 b, int imm) {
		return _mm256_shuffle_pd(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r256i shuffle32(r256i v, int imm) {
		return _mm256_shuffle_epi32(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r256i shuffleHigh16(r256i v, int imm) {
		return _mm256_shufflehi_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r256i shuffleLow16(r256i v, int imm) {
		return _mm256_shufflelo_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r256i shuffleBytes(r256i v, r256i mask) {
		return _mm256_shuffle_epi8(v, mask);
	}

	// integer unpack (low)

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256i unpackLow8(r256i a, r256i b) { return _mm256_unpacklo_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256i unpackLow16(r256i a, r256i b) { return _mm256_unpacklo_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256i unpackLow32(r256i a, r256i b) { return _mm256_unpacklo_epi32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256i unpackLow64(r256i a, r256i b) { return _mm256_unpacklo_epi64(a, b); }

	// integer unpack (high)

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256i unpackHigh8(r256i a, r256i b) { return _mm256_unpackhi_epi8(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256i unpackHigh16(r256i a, r256i b) { return _mm256_unpackhi_epi16(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256i unpackHigh32(r256i a, r256i b) { return _mm256_unpackhi_epi32(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256i unpackHigh64(r256i a, r256i b) { return _mm256_unpackhi_epi64(a, b); }

	// float unpack

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256fp32 unpackLow(r256fp32 a, r256fp32 b) { return _mm256_unpacklo_ps(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256fp32 unpackHigh(r256fp32 a, r256fp32 b) { return _mm256_unpackhi_ps(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256fp64 unpackLow(r256fp64 a, r256fp64 b) { return _mm256_unpacklo_pd(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r256fp64 unpackHigh(r256fp64 a, r256fp64 b) { return _mm256_unpackhi_pd(a, b); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r256i blend16(r256i a, r256i b, int imm) {
		return _mm256_blend_epi16(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r256i blend32(r256i a, r256i b, int imm) {
		return _mm256_blend_epi32(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r256fp32 blend(r256fp32 a, r256fp32 b, int imm) {
		return _mm256_blend_ps(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r256fp64 blend(r256fp64 a, r256fp64 b, int imm) {
		return _mm256_blend_pd(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r256i blendMasked(r256i a, r256i b, r256i mask) {
		return _mm256_blendv_epi8(a, b, mask);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r256fp32 blendMasked(r256fp32 a, r256fp32 b, r256fp32 mask) {
		return _mm256_blendv_ps(a, b, mask);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r256fp64 blendMasked(r256fp64 a, r256fp64 b, r256fp64 mask) {
		return _mm256_blendv_pd(a, b, mask);
	}

	// variable-index permutes

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r128f32 permuteVar(r128f32 v, r128i idx) {
		return _mm_permutevar_ps(v, idx);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r128f64 permuteVar(r128f64 v, r128i idx) {
		return _mm_permutevar_pd(v, idx);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r256fp32 permuteVar(r256fp32 v, r256i idx) {
		return _mm256_permutevar_ps(v, idx);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r256fp64 permuteVar(r256fp64 v, r256i idx) {
		return _mm256_permutevar_pd(v, idx);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r256i permuteVar8x32(r256i v, r256i idx) {
		return _mm256_permutevar8x32_epi32(v, idx);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r256fp32 permuteVar8x32(r256fp32 v, r256i idx) {
		return _mm256_permutevar8x32_ps(v, idx);
	}

	// immediate permutes

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r128f32 permute(r128f32 v, int imm) {
		return _mm_permute_ps(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r128f64 permute(r128f64 v, int imm) {
		return _mm_permute_pd(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r256fp32 permute(r256fp32 v, int imm) {
		return _mm256_permute_ps(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector permute results must not be discarded")
		r256fp64 permute(r256fp64 v, int imm) {
		return _mm256_permute_pd(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane permute results must not be discarded")
		r256fp32 permute2x128(r256fp32 a, r256fp32 b, int imm) {
		return _mm256_permute2f128_ps(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane permute results must not be discarded")
		r256fp64 permute2x128(r256fp64 a, r256fp64 b, int imm) {
		return _mm256_permute2f128_pd(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane permute results must not be discarded")
		r256i permute2x128(r256i a, r256i b, int imm) {
		return _mm256_permute2f128_si256(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane permute results must not be discarded")
		r256i permute4x64(r256i v, int imm) {
		return _mm256_permute4x64_epi64(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane permute results must not be discarded")
		r256fp64 permute4x64(r256fp64 v, int imm) {
		return _mm256_permute4x64_pd(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane permute results must not be discarded")
		r256i permute2x128(r256i v, int imm) {
		return _mm256_permute2x128_si256(v, v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane insertion results must not be discarded")
		r256fp32 insert128(r256fp32 v, r128f32 lane, int imm) {
		return _mm256_insertf128_ps(v, lane, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane insertion results must not be discarded")
		r256fp64 insert128(r256fp64 v, r128f64 lane, int imm) {
		return _mm256_insertf128_pd(v, lane, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane insertion results must not be discarded")
		r256i insert128(r256i v, r128i lane, int imm) {
		return _mm256_insertf128_si256(v, lane, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane extraction results must not be discarded")
		r128f32 extract128(r256fp32 v, int imm) {
		return _mm256_extractf128_ps(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane extraction results must not be discarded")
		r128f64 extract128(r256fp64 v, int imm) {
		return _mm256_extractf128_pd(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane extraction results must not be discarded")
		r128i extract128(r256i v, int imm) {
		return _mm256_extractf128_si256(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane insertion results must not be discarded")
		r256i insertI128(r256i v, r128i lane, int imm) {
		return _mm256_inserti128_si256(v, lane, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector lane extraction results must not be discarded")
		r128i extractI128(r256i v, int imm) {
		return _mm256_extracti128_si256(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector alignment results must not be discarded")
		r256i alignRightBytes(r256i a, r256i b, int imm) {
		return _mm256_alignr_epi8(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256fp32 hadd(r256fp32 a, r256fp32 b) {
		return _mm256_hadd_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256fp32 hsub(r256fp32 a, r256fp32 b) {
		return _mm256_hsub_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256fp64 hadd(r256fp64 a, r256fp64 b) {
		return _mm256_hadd_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256fp64 hsub(r256fp64 a, r256fp64 b) {
		return _mm256_hsub_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256i hadd16(r256i a, r256i b) {
		return _mm256_hadd_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256i hadd32(r256i a, r256i b) {
		return _mm256_hadd_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256i hsub16(r256i a, r256i b) {
		return _mm256_hsub_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256i hsub32(r256i a, r256i b) {
		return _mm256_hsub_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256i haddSaturate16(r256i a, r256i b) {
		return _mm256_hadds_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256i hsubSaturate16(r256i a, r256i b) {
		return _mm256_hsubs_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256i sumAbsDiffU8(r256i a, r256i b) {
		return _mm256_sad_epu8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r256i mpsadBW(r256i a, r256i b, int imm) {
		return _mm256_mpsadbw_epu8(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector sign results must not be discarded")
		r256i sign8(r256i a, r256i b) {
		return _mm256_sign_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector sign results must not be discarded")
		r256i sign16(r256i a, r256i b) {
		return _mm256_sign_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector sign results must not be discarded")
		r256i sign32(r256i a, r256i b) {
		return _mm256_sign_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector absolute value results must not be discarded")
		r256i abs8(r256i v) {
		return _mm256_abs_epi8(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector absolute value results must not be discarded")
		r256i abs16(r256i v) {
		return _mm256_abs_epi16(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector absolute value results must not be discarded")
		r256i abs32(r256i v) {
		return _mm256_abs_epi32(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector math results must not be discarded")
		r256fp32 sqrt(r256fp32 v) {
		return _mm256_sqrt_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector math results must not be discarded")
		r256fp64 sqrt(r256fp64 v) {
		return _mm256_sqrt_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector math results must not be discarded")
		r256fp32 rsqrt(r256fp32 v) {
		return _mm256_rsqrt_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector math results must not be discarded")
		r256fp32 rcp(r256fp32 v) {
		return _mm256_rcp_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r256fp32 round(r256fp32 v, int imm) {
		return _mm256_round_ps(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r256fp64 round(r256fp64 v, int imm) {
		return _mm256_round_pd(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r256fp32 ceil(r256fp32 v) {
		return _mm256_ceil_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r256fp64 ceil(r256fp64 v) {
		return _mm256_ceil_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r256fp32 floor(r256fp32 v) {
		return _mm256_floor_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r256fp64 floor(r256fp64 v) {
		return _mm256_floor_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector packing results must not be discarded")
		r256i pack16(r256i a, r256i b) {
		return _mm256_packs_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector packing results must not be discarded")
		r256i pack32(r256i a, r256i b) {
		return _mm256_packs_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector packing results must not be discarded")
		r256i packUnsigned16(r256i a, r256i b) {
		return _mm256_packus_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector packing results must not be discarded")
		r256i packUnsigned32(r256i a, r256i b) {
		return _mm256_packus_epi32(a, b);
	}

	// integer broadcasts (from low lane)

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r128i broadcast8(r128i v) { return _mm_broadcastb_epi8(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r128i broadcast16(r128i v) { return _mm_broadcastw_epi16(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r128i broadcast32(r128i v) { return _mm_broadcastd_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r128i broadcast64(r128i v) { return _mm_broadcastq_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r256i broadcast8avx2(r128i v) { return _mm256_broadcastb_epi8(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r256i broadcast16avx2(r128i v) { return _mm256_broadcastw_epi16(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r256i broadcast32avx2(r128i v) { return _mm256_broadcastd_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r256i broadcast64avx2(r128i v) { return _mm256_broadcastq_epi64(v); }

	// floating-point broadcasts

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r128f32 broadcastScalar(float const* p) { return _mm_broadcast_ss(p); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r256fp32 broadcastScalar256(float const* p) { return _mm256_broadcast_ss(p); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r256fp64 broadcastScalar256(double const* p) { return _mm256_broadcast_sd(p); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r256fp32 broadcast(r128f32 const* v) { return _mm256_broadcast_ps(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector broadcast results must not be discarded")
		r256fp64 broadcast(r128f64 const* v) { return _mm256_broadcast_pd(v); }

	// float <-> double

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256fp64 cvtFp32ToFp64(r128f32 v) { return _mm256_cvtps_pd(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128f32 cvtFp64ToFp32(r256fp64 v) { return _mm256_cvtpd_ps(v); }

	// float -> int

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtFp32ToInt32(r256fp32 v) { return _mm256_cvtps_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtFp32ToInt32Trunc(r256fp32 v) { return _mm256_cvttps_epi32(v); }

	// int -> float / double

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256fp32 cvtInt32ToFp32(r256i v) { return _mm256_cvtepi32_ps(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256fp64 cvtInt32ToFp64(r128i v) { return _mm256_cvtepi32_pd(v); }

	// double -> int

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i cvtFp64ToInt32(r256fp64 v) { return _mm256_cvtpd_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i cvtFp64ToInt32Trunc(r256fp64 v) { return _mm256_cvttpd_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtI8ToI16(r128i v) { return _mm256_cvtepi8_epi16(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtI8ToI32(r128i v) { return _mm256_cvtepi8_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtI8ToI64(r128i v) { return _mm256_cvtepi8_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtI16ToI32(r128i v) { return _mm256_cvtepi16_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtI16ToI64(r128i v) { return _mm256_cvtepi16_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtI32ToI64(r128i v) { return _mm256_cvtepi32_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtU8ToU16(r128i v) { return _mm256_cvtepu8_epi16(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtU8ToU32(r128i v) { return _mm256_cvtepu8_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtU8ToU64(r128i v) { return _mm256_cvtepu8_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtU16ToU32(r128i v) { return _mm256_cvtepu16_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtU16ToU64(r128i v) { return _mm256_cvtepu16_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r256i cvtU32ToU64(r128i v) { return _mm256_cvtepu32_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256fp64 castFp32ToFp64(r256fp32 v) { return _mm256_castps_pd(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256fp32 castFp64ToFp32(r256fp64 v) { return _mm256_castpd_ps(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256i castFp32ToInt(r256fp32 v) { return _mm256_castps_si256(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256fp32 castIntToFp32(r256i v) { return _mm256_castsi256_ps(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256i castFp64ToInt(r256fp64 v) { return _mm256_castpd_si256(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256fp64 castIntToFp64(r256i v) { return _mm256_castsi256_pd(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r128f32 cast256To128(r256fp32 v) { return _mm256_castps256_ps128(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r128f64 cast256To128(r256fp64 v) { return _mm256_castpd256_pd128(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r128i cast256To128(r256i v) { return _mm256_castsi256_si128(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256fp32 cast128To256(r128f32 v) { return _mm256_castps128_ps256(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256fp64 cast128To256(r128f64 v) { return _mm256_castpd128_pd256(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector cast results must not be discarded")
		r256i cast128To256(r128i v) { return _mm256_castsi128_si256(v); }

	// NOTE: scale is in bytes and must be 1, 2, 4, or 8

	STL_FORCEINLINE
		STL_NODISCARD("Vector gather results must not be discarded")
		r128f32 gatherFp32(const float* base, r128i idx, int scale) {
		return _mm_i32gather_ps(base, idx, scale);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector gather results must not be discarded")
		r256fp32 gatherFp32(const float* base, r256i idx, int scale) {
		return _mm256_i32gather_ps(base, idx, scale);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector gather results must not be discarded")
		r256fp64 gatherFp64(const double* base, r128i idx, int scale) {
		return _mm256_i32gather_pd(base, idx, scale);
	}

	// masked example (pattern repeats)

	STL_FORCEINLINE
		STL_NODISCARD("Vector gather results must not be discarded")
		r256fp32 gatherMasked(const float* base, r256i idx, r256fp32 mask, int scale) {
		return _mm256_mask_i32gather_ps(
			_mm256_setzero_ps(), base, idx, mask, scale
		);
	}

	// NOTE: streaming and utility ops intentionally return void

	STL_FORCEINLINE
		void stream(float* p, r256fp32 v) { _mm256_stream_ps(p, v); }

	STL_FORCEINLINE
		void stream(double* p, r256fp64 v) { _mm256_stream_pd(p, v); }

	STL_FORCEINLINE
		void stream(r256i* p, r256i v) { _mm256_stream_si256(p, v); }

	STL_FORCEINLINE void zeroAll() { _mm256_zeroall(); }
	STL_FORCEINLINE void zeroUpper() { _mm256_zeroupper(); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp32 set(r128f32 hi, r128f32 lo) { return _mm256_set_m128(hi, lo); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp64 set(r128f64 hi, r128f64 lo) { return _mm256_set_m128d(hi, lo); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i set(r128i hi, r128i lo) { return _mm256_set_m128i(hi, lo); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp32 setReversed(r128f32 lo, r128f32 hi) { return _mm256_setr_m128(lo, hi); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256fp64 setReversed(r128f64 lo, r128f64 hi) { return _mm256_setr_m128d(lo, hi); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector construction results must not be discarded")
		r256i setReversed(r128i lo, r128i hi) { return _mm256_setr_m128i(lo, hi); }
}
#endif
#endif


