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

#ifndef ALLOW_VECTOR_INTRIN_128 
#error "Explicitly define ALLOW_VECTOR_INTRIN_128 before including this header."
#else
#if !defined(__SSE__) && !defined(_vectorbackend_EDITOR_MODE)
#error "_vectorbackend SIMD Vector Backend requires AVX2 flag to be enabled during compilation"
#else
namespace StormSTL::Vectorization::Intrinsic::v128 {
	using r128i = __m128i;
	using r128f32 = __m128;
	using r128f64 = __m128d;

	// SSE: Vector construction and constants (float32)
	// SSE2: Vector construction (float64)
	// SSE2: Vector construction (integer / __m128i)
	//
	// SSE: Vector load and store (float32)
	// SSE2: Vector load and store (float64)
	// SSE2: Vector load and store (integer)
	//
	// SSE: Partial vector load and store (float32, 64-bit halves)
	// SSE2: Scalar load and store (float64, lower lane only)
	//
	// SSE: Basic arithmetic (float32)
	// SSE2: Basic arithmetic (float64)
	// SSE2: Integer arithmetic (add/sub, wraparound and saturation)
	//
	// SSE2: Integer multiply (16-bit lanes, widening variants)
	// SSE4.1: Integer multiply (32-bit lanes)
	// SSE2 / SSSE3: Pairwise multiply-add integer arithmetic
	//
	// SSE: Min/max (float32)
	// SSE2: Min/max (float64)
	// SSE2: Min/max (integer, limited set)
	// SSE4.1: Min/max (extended integer set)
	//
	// SSE: Vector comparisons (float32, full-lane mask results)
	//
	// SSE2: Vector comparisons (float64, full-lane mask results, IEEE-754 semantics)
	// SSE2: Vector comparisons (integer equality and greater-than, 8/16/32-bit)
	// SSE4.1: Vector comparisons (integer equality, 64-bit)
	// SSE4.2: Vector comparisons (integer greater-than, 64-bit)
	//
	// SSE: Predicate extraction (float32 movemask)
	// SSE2: Predicate extraction (float64 and integer movemask)
	// SSE4.1: Vector test operations (all-zero / all-one / mixed bits)
	//
	// SSE: Bitwise operations (float32)
	// SSE2: Bitwise operations (float64)
	// SSE2: Bitwise operations (integer / __m128i)
	//
	// SSE2: Vector shifts (integer)
	// immediate shifts (compile-time constant counts)
	// variable shifts (count taken from vector register)
	// logical left/right and arithmetic right semantics
	//
	// SSE: Vector shuffle (float32 / float64, immediate control masks)
	// SSE2: Vector shuffle (integer lane reordering)
	// SSSE3: Byte-wise vector shuffle (arbitrary permutation)
	//
	// SSE2: Unpack operations (interleave low/high lanes)
	// integer unpack (8/16/32/64-bit)
	// float32 and float64 unpack (lane interleaving)
	//
	// SSE4.1: Blend operations (immediate and mask-based selection)
	// lane-wise selection between two vectors
	//
	// SSE2 / SSE4.1: Insert and extract operations
	// scalar <-> vector lane movement for integers and floats
	//
	// SSSE3: Byte alignment and lane concatenation
	//
	// SSE3 / SSSE3: Horizontal operations
	// horizontal add/sub, sum-of-absolute-differences, saturated variants
	//
	// SSE4.1: Dot product operations (masked lane accumulation)
	//
	// SSSE3: Sign control operations (per-lane sign application)
	// SSSE3: Absolute value operations (per-lane)
	//
	// SSE: Square root and reciprocal operations
	// scalar and vector, approximate and exact variants
	//
	// SSE4.1: Rounding operations
	// round, ceil, floor with immediate control
	//
	// SSE2: Packing operations
	// narrowing with signed/unsigned saturation
	//
	// SSE2 / SSE4.1: Type conversion operations
	// float <-> double, float/double <-> integer, scalar extraction/insertion
	// signed and unsigned integer widening conversions
	//
	// SSE2: Bitwise casts
	// zero-cost reinterpretation between float, double, and integer vectors
	//
	// FMA (FMA3): 128-bit fused multiply-add operations
	// single-rounding multiply-add/subtract variants
	//
	// SSE / SSE2: Additional integer test operations
	// bitwise zero / containment / mixed-bit tests
	//
	// SSE / SSE2: Streaming and memory operations
	// non-temporal stores, cache control, fences, spin-wait
	// scalar <-> vector conversions for float32 / float64
	//
	// SSE: Scalar extraction
	// extract lower scalar from vector register



	STL_FORCEINLINE STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128f32 zeroSinglePrecision() {
		return _mm_setzero_ps();
	}

	STL_FORCEINLINE STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128f32 setOne(float v_Value) {
		return _mm_set1_ps(v_Value);
	}

	STL_FORCEINLINE STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128f32 setReversed(float v_3, float v_2, float v_1, float v_0) {
		return _mm_set_ps(v_3, v_2, v_1, v_0);
	}

	STL_FORCEINLINE STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128f32 set(float v_0, float v_1, float v_2, float v_3) {
		return _mm_setr_ps(v_0, v_1, v_2, v_3);
	}

	STL_FORCEINLINE STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128f64 zeroDoublePrecision() {
		return _mm_setzero_pd();
	}

	STL_FORCEINLINE STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128f64 setOne(double v_Value) {
		return _mm_set1_pd(v_Value);
	}

	STL_FORCEINLINE STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128f64 setReversed(double v_1, double v_0) {
		return _mm_set_pd(v_1, v_0);
	}

	STL_FORCEINLINE STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128f64 set(double v_0, double v_1) {
		return _mm_setr_pd(v_0, v_1);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i zero() {
		return _mm_setzero_si128();
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i setOne(int8_t v_Value) {
		return _mm_set1_epi8(v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i setOne(int16_t v_Value) {
		return _mm_set1_epi16(v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i setOne(int32_t v_Value) {
		return _mm_set1_epi32(v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i setOne(int64_t v_Value) {
		return _mm_set1_epi64x(v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i setReversed(
			int8_t  v_15, int8_t  v_14, int8_t  v_13, int8_t  v_12,
			int8_t  v_11, int8_t  v_10, int8_t  v_9, int8_t  v_8,
			int8_t  v_7, int8_t  v_6, int8_t  v_5, int8_t  v_4,
			int8_t  v_3, int8_t  v_2, int8_t  v_1, int8_t  v_0
		) {
		return _mm_set_epi8(
			v_15, v_14, v_13, v_12, v_11, v_10, v_9, v_8,
			v_7, v_6, v_5, v_4, v_3, v_2, v_1, v_0
		);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i setReversed(
			int16_t v_7, int16_t v_6, int16_t v_5, int16_t v_4,
			int16_t v_3, int16_t v_2, int16_t v_1, int16_t v_0
		) {
		return _mm_set_epi16(
			v_7, v_6, v_5, v_4,
			v_3, v_2, v_1, v_0
		);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i setReversed(
			int32_t v_3, int32_t v_2, int32_t v_1, int32_t v_0
		) {
		return _mm_set_epi32(v_3, v_2, v_1, v_0);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i setReversed(
			int64_t v_1, int64_t v_0
		) {
		return _mm_set_epi64x(v_1, v_0);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i set(
			int8_t  v_0, int8_t  v_1, int8_t  v_2, int8_t  v_3,
			int8_t  v_4, int8_t  v_5, int8_t  v_6, int8_t  v_7,
			int8_t  v_8, int8_t  v_9, int8_t  v_10, int8_t  v_11,
			int8_t  v_12, int8_t  v_13, int8_t  v_14, int8_t  v_15
		) {
		return _mm_setr_epi8(
			v_0, v_1, v_2, v_3,
			v_4, v_5, v_6, v_7,
			v_8, v_9, v_10, v_11,
			v_12, v_13, v_14, v_15
		);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i set(
			int16_t v_0, int16_t v_1, int16_t v_2, int16_t v_3,
			int16_t v_4, int16_t v_5, int16_t v_6, int16_t v_7
		) {
		return _mm_setr_epi16(
			v_0, v_1, v_2, v_3,
			v_4, v_5, v_6, v_7
		);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i set(
			int32_t v_0, int32_t v_1, int32_t v_2, int32_t v_3
		) {
		return _mm_setr_epi32(v_0, v_1, v_2, v_3);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector functions cannot have their return values discarded")
		r128i set(
			int64_t v_0, int64_t v_1
		) {
		return _mm_setr_epi64x(v_0, v_1);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128f32 loadAligned(const float* p_Address) {
		return _mm_load_ps(p_Address); // requires 16-byte alignment
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128f32 loadUnaligned(const float* p_Address) {
		return _mm_loadu_ps(p_Address);
	}

	STL_FORCEINLINE
		void storeAligned(float* p_Address, r128f32 v_Value) {
		_mm_store_ps(p_Address, v_Value); // requires 16-byte alignment
	}

	STL_FORCEINLINE
		void storeUnaligned(float* p_Address, r128f32 v_Value) {
		_mm_storeu_ps(p_Address, v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128f64 loadAligned(const double* p_Address) {
		return _mm_load_pd(p_Address); // requires 16-byte alignment
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128f64 loadUnaligned(const double* p_Address) {
		return _mm_loadu_pd(p_Address);
	}

	STL_FORCEINLINE
		void storeAligned(double* p_Address, r128f64 v_Value) {
		_mm_store_pd(p_Address, v_Value); // requires 16-byte alignment
	}

	STL_FORCEINLINE
		void storeUnaligned(double* p_Address, r128f64 v_Value) {
		_mm_storeu_pd(p_Address, v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128i loadAligned(const r128i* p_Address) {
		return _mm_load_si128(p_Address); // requires 16-byte alignment
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128i loadUnaligned(const r128i* p_Address) {
		return _mm_loadu_si128(p_Address);
	}

	STL_FORCEINLINE
		void storeAligned(r128i* p_Address, r128i v_Value) {
		_mm_store_si128(p_Address, v_Value); // requires 16-byte alignment
	}

	STL_FORCEINLINE
		void storeUnaligned(r128i* p_Address, r128i v_Value) {
		_mm_storeu_si128(p_Address, v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128f32 loadLow(const r128f32 v_Base, const float* p_Address) {
		return _mm_loadl_pi(v_Base, reinterpret_cast<const __m64*>(p_Address));
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128f32 loadHigh(const r128f32 v_Base, const float* p_Address) {
		return _mm_loadh_pi(v_Base, reinterpret_cast<const __m64*>(p_Address));
	}

	STL_FORCEINLINE
		void storeLow(float* p_Address, r128f32 v_Value) {
		_mm_storel_pi(reinterpret_cast<__m64*>(p_Address), v_Value);
	}

	STL_FORCEINLINE
		void storeHigh(float* p_Address, r128f32 v_Value) {
		_mm_storeh_pi(reinterpret_cast<__m64*>(p_Address), v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector loads cannot have their return values discarded")
		r128f64 loadScalar(const double* p_Address) {
		return _mm_load_sd(p_Address);
	}

	STL_FORCEINLINE
		void storeScalar(double* p_Address, r128f64 v_Value) {
		_mm_store_sd(p_Address, v_Value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f32 add(r128f32 a, r128f32 b) {
		return _mm_add_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f32 sub(r128f32 a, r128f32 b) {
		return _mm_sub_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f32 mul(r128f32 a, r128f32 b) {
		return _mm_mul_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f32 div(r128f32 a, r128f32 b) {
		return _mm_div_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f64 add(r128f64 a, r128f64 b) {
		return _mm_add_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f64 sub(r128f64 a, r128f64 b) {
		return _mm_sub_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f64 mul(r128f64 a, r128f64 b) {
		return _mm_mul_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f64 div(r128f64 a, r128f64 b) {
		return _mm_div_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i add(r128i a, r128i b) {
		return _mm_add_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i add16(r128i a, r128i b) {
		return _mm_add_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i add32(r128i a, r128i b) {
		return _mm_add_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i add64(r128i a, r128i b) {
		return _mm_add_epi64(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i sub(r128i a, r128i b) {
		return _mm_sub_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i sub16(r128i a, r128i b) {
		return _mm_sub_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i sub32(r128i a, r128i b) {
		return _mm_sub_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i sub64(r128i a, r128i b) {
		return _mm_sub_epi64(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i addSaturate(r128i a, r128i b) {
		return _mm_adds_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i addSaturate16(r128i a, r128i b) {
		return _mm_adds_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i subSaturate(r128i a, r128i b) {
		return _mm_subs_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i subSaturate16(r128i a, r128i b) {
		return _mm_subs_epi16(a, b);
	}


	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i addSaturateUnsigned(r128i a, r128i b) {
		return _mm_adds_epu8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i addSaturateUnsigned16(r128i a, r128i b) {
		return _mm_adds_epu16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i subSaturateUnsigned(r128i a, r128i b) {
		return _mm_subs_epu8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i subSaturateUnsigned16(r128i a, r128i b) {
		return _mm_subs_epu16(a, b);
	}


// low 16 bits of signed multiplication
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i mulLow16(r128i a, r128i b) {
		return _mm_mullo_epi16(a, b);
	}

	// high 16 bits of signed multiplication
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i mulHigh16(r128i a, r128i b) {
		return _mm_mulhi_epi16(a, b);
	}

	// high 16 bits of unsigned multiplication
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i mulHighUnsigned16(r128i a, r128i b) {
		return _mm_mulhi_epu16(a, b);
	}


	// multiplies even 32-bit lanes, returns full 64-bit results
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i mulUnsigned32To64(r128i a, r128i b) {
		return _mm_mul_epu32(a, b);
	}

	// low 32 bits of signed 32-bit multiplication
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i mulLow32(r128i a, r128i b) {
		return _mm_mullo_epi32(a, b);
	}

	// (a0*b0 + a1*b1), producing 32-bit results per pair
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i mulAdd16(r128i a, r128i b) {
		return _mm_madd_epi16(a, b);
	}

	// (unsigned byte x signed byte) pairwise multiply-add -> int16
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i mulAddUnsignedByteSignedByte(r128i a, r128i b) {
		return _mm_maddubs_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f32 min(r128f32 a, r128f32 b) {
		return _mm_min_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f32 max(r128f32 a, r128f32 b) {
		return _mm_max_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f64 min(r128f64 a, r128f64 b) {
		return _mm_min_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128f64 max(r128f64 a, r128f64 b) {
		return _mm_max_pd(a, b);
	}

	// signed 16-bit
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i min16(r128i a, r128i b) {
		return _mm_min_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i max16(r128i a, r128i b) {
		return _mm_max_epi16(a, b);
	}

	// unsigned 8-bit
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i minUnsigned8(r128i a, r128i b) {
		return _mm_min_epu8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i maxUnsigned8(r128i a, r128i b) {
		return _mm_max_epu8(a, b);
	}

	// signed 8-bit
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i min8(r128i a, r128i b) {
		return _mm_min_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i max8(r128i a, r128i b) {
		return _mm_max_epi8(a, b);
	}

	// signed 32-bit
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i min32(r128i a, r128i b) {
		return _mm_min_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i max32(r128i a, r128i b) {
		return _mm_max_epi32(a, b);
	}

	// unsigned 16-bit
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i minUnsigned16(r128i a, r128i b) {
		return _mm_min_epu16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i maxUnsigned16(r128i a, r128i b) {
		return _mm_max_epu16(a, b);
	}

	// unsigned 32-bit
	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i minUnsigned32(r128i a, r128i b) {
		return _mm_min_epu32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector arithmetic results must not be discarded")
		r128i maxUnsigned32(r128i a, r128i b) {
		return _mm_max_epu32(a, b);
	}


	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpEqual(r128f32 a, r128f32 b) {
		return _mm_cmpeq_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpNotEqual(r128f32 a, r128f32 b) {
		return _mm_cmpneq_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpLess(r128f32 a, r128f32 b) {
		return _mm_cmplt_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpLessEqual(r128f32 a, r128f32 b) {
		return _mm_cmple_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpGreater(r128f32 a, r128f32 b) {
		return _mm_cmpgt_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpGreaterEqual(r128f32 a, r128f32 b) {
		return _mm_cmpge_ps(a, b);
	}


	// true where neither operand is NaN
	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpOrdered(r128f32 a, r128f32 b) {
		return _mm_cmpord_ps(a, b);
	}

	// true where either operand is NaN
	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpUnordered(r128f32 a, r128f32 b) {
		return _mm_cmpunord_ps(a, b);
	}	

	// !(a < b) OR unordered
	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpNotLess(r128f32 a, r128f32 b) {
		return _mm_cmpnlt_ps(a, b);
	}

	// !(a <= b) OR unordered
	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpNotLessEqual(r128f32 a, r128f32 b) {
		return _mm_cmpnle_ps(a, b);
	}

	// !(a > b) OR unordered
	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpNotGreater(r128f32 a, r128f32 b) {
		return _mm_cmpngt_ps(a, b);
	}

	// !(a >= b) OR unordered
	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f32 cmpNotGreaterEqual(r128f32 a, r128f32 b) {
		return _mm_cmpnge_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpEqual(r128f64 a, r128f64 b) {
		return _mm_cmpeq_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpNotEqual(r128f64 a, r128f64 b) {
		return _mm_cmpneq_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpLess(r128f64 a, r128f64 b) {
		return _mm_cmplt_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpLessEqual(r128f64 a, r128f64 b) {
		return _mm_cmple_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpGreater(r128f64 a, r128f64 b) {
		return _mm_cmpgt_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpGreaterEqual(r128f64 a, r128f64 b) {
		return _mm_cmpge_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpOrdered(r128f64 a, r128f64 b) {
		return _mm_cmpord_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpUnordered(r128f64 a, r128f64 b) {
		return _mm_cmpunord_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpNotLess(r128f64 a, r128f64 b) {
		return _mm_cmpnlt_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpNotLessEqual(r128f64 a, r128f64 b) {
		return _mm_cmpnle_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpNotGreater(r128f64 a, r128f64 b) {
		return _mm_cmpngt_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128f64 cmpNotGreaterEqual(r128f64 a, r128f64 b) {
		return _mm_cmpnge_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128i cmpEqual8(r128i a, r128i b) {
		return _mm_cmpeq_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128i cmpEqual16(r128i a, r128i b) {
		return _mm_cmpeq_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128i cmpEqual32(r128i a, r128i b) {
		return _mm_cmpeq_epi32(a, b);
	}

	// SSE4.1
	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128i cmpEqual64(r128i a, r128i b) {
		return _mm_cmpeq_epi64(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128i cmpGreater8(r128i a, r128i b) {
		return _mm_cmpgt_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128i cmpGreater16(r128i a, r128i b) {
		return _mm_cmpgt_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128i cmpGreater32(r128i a, r128i b) {
		return _mm_cmpgt_epi32(a, b);
	}

	// SSE4.2
	STL_FORCEINLINE
		STL_NODISCARD("Vector comparison results must not be discarded")
		r128i cmpGreater64(r128i a, r128i b) {
		return _mm_cmpgt_epi64(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Predicate extraction results must not be discarded")
		int movemask(r128f32 v) {
		return _mm_movemask_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Predicate extraction results must not be discarded")
		int movemask(r128f64 v) {
		return _mm_movemask_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Predicate extraction results must not be discarded")
		int movemask(r128i v) {
		return _mm_movemask_epi8(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testAllZero(r128i a, r128i mask) {
		return _mm_test_all_zeros(a, mask);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testAllOnes(r128i a) {
		return _mm_test_all_ones(a);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector test results must not be discarded")
		bool testMixedOnesZeros(r128i a, r128i mask) {
		return _mm_test_mix_ones_zeros(a, mask);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128f32 bitAnd(r128f32 a, r128f32 b) {
		return _mm_and_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128f32 bitOr(r128f32 a, r128f32 b) {
		return _mm_or_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128f32 bitXor(r128f32 a, r128f32 b) {
		return _mm_xor_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128f32 bitAndNot(r128f32 a, r128f32 b) {
		return _mm_andnot_ps(a, b); // (~a) & b
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128f64 bitAnd(r128f64 a, r128f64 b) {
		return _mm_and_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128f64 bitOr(r128f64 a, r128f64 b) {
		return _mm_or_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128f64 bitXor(r128f64 a, r128f64 b) {
		return _mm_xor_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128f64 bitAndNot(r128f64 a, r128f64 b) {
		return _mm_andnot_pd(a, b); // (~a) & b
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128i bitAnd(r128i a, r128i b) {
		return _mm_and_si128(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128i bitOr(r128i a, r128i b) {
		return _mm_or_si128(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128i bitXor(r128i a, r128i b) {
		return _mm_xor_si128(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector bitwise results must not be discarded")
		r128i bitAndNot(r128i a, r128i b) {
		return _mm_andnot_si128(a, b); // (~a) & b
	}


// logical left shift

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftLeft16(r128i v, int imm) {
		return _mm_slli_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftLeft32(r128i v, int imm) {
		return _mm_slli_epi32(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftLeft64(r128i v, int imm) {
		return _mm_slli_epi64(v, imm);
	}

	// logical right shift

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightLogical16(r128i v, int imm) {
		return _mm_srli_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightLogical32(r128i v, int imm) {
		return _mm_srli_epi32(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightLogical64(r128i v, int imm) {
		return _mm_srli_epi64(v, imm);
	}

	// arithmetic right shift (sign-extending)

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightArithmetic16(r128i v, int imm) {
		return _mm_srai_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightArithmetic32(r128i v, int imm) {
		return _mm_srai_epi32(v, imm);
	}


	// logical left shift

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftLeft16(r128i v, r128i count) {
		return _mm_sll_epi16(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftLeft32(r128i v, r128i count) {
		return _mm_sll_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftLeft64(r128i v, r128i count) {
		return _mm_sll_epi64(v, count);
	}

	// logical right shift

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightLogical16(r128i v, r128i count) {
		return _mm_srl_epi16(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightLogical32(r128i v, r128i count) {
		return _mm_srl_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightLogical64(r128i v, r128i count) {
		return _mm_srl_epi64(v, count);
	}

	// arithmetic right shift

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightArithmetic16(r128i v, r128i count) {
		return _mm_sra_epi16(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shift results must not be discarded")
		r128i shiftRightArithmetic32(r128i v, r128i count) {
		return _mm_sra_epi32(v, count);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r128f32 shuffle(r128f32 a, r128f32 b, int imm) {
		return _mm_shuffle_ps(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r128f64 shuffle(r128f64 a, r128f64 b, int imm) {
		return _mm_shuffle_pd(a, b, imm);
	}

	// SSE2 - Shuffle (Integer)

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r128i shuffleHigh16(r128i v, int imm) {
		return _mm_shufflehi_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r128i shuffleLow16(r128i v, int imm) {
		return _mm_shufflelo_epi16(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r128i shuffle32(r128i v, int imm) {
		return _mm_shuffle_epi32(v, imm);
	}

	// SSSE3 - Byte-wise Shuffle

	STL_FORCEINLINE
		STL_NODISCARD("Vector shuffle results must not be discarded")
		r128i shuffleBytes(r128i v, r128i mask) {
		return _mm_shuffle_epi8(v, mask);
	}

	// Integer unpack - lower lanes

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128i unpackLow8(r128i a, r128i b) {
		return _mm_unpacklo_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128i unpackLow16(r128i a, r128i b) {
		return _mm_unpacklo_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128i unpackLow32(r128i a, r128i b) {
		return _mm_unpacklo_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128i unpackLow64(r128i a, r128i b) {
		return _mm_unpacklo_epi64(a, b);
	}

	// Integer unpack - higher lanes

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128i unpackHigh8(r128i a, r128i b) {
		return _mm_unpackhi_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128i unpackHigh16(r128i a, r128i b) {
		return _mm_unpackhi_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128i unpackHigh32(r128i a, r128i b) {
		return _mm_unpackhi_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128i unpackHigh64(r128i a, r128i b) {
		return _mm_unpackhi_epi64(a, b);
	}

	// Float32 unpack (interleave)

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128f32 unpackLow(r128f32 a, r128f32 b) {
		return _mm_unpacklo_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128f32 unpackHigh(r128f32 a, r128f32 b) {
		return _mm_unpackhi_ps(a, b);
	}

	// Float64 unpack (interleave)

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128f64 unpackLow(r128f64 a, r128f64 b) {
		return _mm_unpacklo_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector unpack results must not be discarded")
		r128f64 unpackHigh(r128f64 a, r128f64 b) {
		return _mm_unpackhi_pd(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r128i blend16(r128i a, r128i b, int imm) {
		return _mm_blend_epi16(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r128f32 blend(r128f32 a, r128f32 b, int imm) {
		return _mm_blend_ps(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r128f64 blend(r128f64 a, r128f64 b, int imm) {
		return _mm_blend_pd(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r128i blendMasked(r128i a, r128i b, r128i mask) {
		return _mm_blendv_epi8(a, b, mask);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r128f32 blendMasked(r128f32 a, r128f32 b, r128f32 mask) {
		return _mm_blendv_ps(a, b, mask);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector blend results must not be discarded")
		r128f64 blendMasked(r128f64 a, r128f64 b, r128f64 mask) {
		return _mm_blendv_pd(a, b, mask);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector insert results must not be discarded")
		r128i insert16(r128i v, int value, int index) {
		return _mm_insert_epi16(v, value, index);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector insert results must not be discarded")
		r128i insert8(r128i v, int value, int index) {
		return _mm_insert_epi8(v, value, index);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector insert results must not be discarded")
		r128i insert32(r128i v, int value, int index) {
		return _mm_insert_epi32(v, value, index);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector insert results must not be discarded")
		r128i insert64(r128i v, long long value, int index) {
		return _mm_insert_epi64(v, value, index);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector extract results must not be discarded")
		int extract16(r128i v, int index) {
		return _mm_extract_epi16(v, index);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector extract results must not be discarded")
		int extract8(r128i v, int index) {
		return _mm_extract_epi8(v, index);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector extract results must not be discarded")
		int extract32(r128i v, int index) {
		return _mm_extract_epi32(v, index);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector extract results must not be discarded")
		long long extract64(r128i v, int index) {
		return _mm_extract_epi64(v, index);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector insert results must not be discarded")
		r128f32 insert(r128f32 v, r128f32 value, int imm) {
		return _mm_insert_ps(v, value, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector extract results must not be discarded")
		int extract(r128f32 v, int imm) {
		return _mm_extract_ps(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector alignment results must not be discarded")
		r128i alignRight(r128i a, r128i b, int imm) {
		return _mm_alignr_epi8(a, b, imm);
	}

	// float32 horizontal add/sub

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r128f32 horizontalAdd(r128f32 a, r128f32 b) {
		return _mm_hadd_ps(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r128f32 horizontalSub(r128f32 a, r128f32 b) {
		return _mm_hsub_ps(a, b);
	}

	// integer horizontal add/sub

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r128i horizontalAdd16(r128i a, r128i b) {
		return _mm_hadd_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r128i horizontalAdd32(r128i a, r128i b) {
		return _mm_hadd_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r128i horizontalSub16(r128i a, r128i b) {
		return _mm_hsub_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r128i horizontalSub32(r128i a, r128i b) {
		return _mm_hsub_epi32(a, b);
	}

	// sum of absolute differences

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r128i sumAbsDiffUnsigned8(r128i a, r128i b) {
		return _mm_sad_epu8(a, b);
	}

	// saturated horizontal add

	STL_FORCEINLINE
		STL_NODISCARD("Vector horizontal results must not be discarded")
		r128i horizontalAddSaturate16(r128i a, r128i b) {
		return _mm_hadds_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector dot product results must not be discarded")
		r128f32 dot(r128f32 a, r128f32 b, int imm) {
		return _mm_dp_ps(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector dot product results must not be discarded")
		r128f64 dot(r128f64 a, r128f64 b, int imm) {
		return _mm_dp_pd(a, b, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector sign results must not be discarded")
		r128i sign8(r128i a, r128i b) {
		return _mm_sign_epi8(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector sign results must not be discarded")
		r128i sign16(r128i a, r128i b) {
		return _mm_sign_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector sign results must not be discarded")
		r128i sign32(r128i a, r128i b) {
		return _mm_sign_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector absolute value results must not be discarded")
		r128i abs8(r128i v) {
		return _mm_abs_epi8(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector absolute value results must not be discarded")
		r128i abs16(r128i v) {
		return _mm_abs_epi16(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector absolute value results must not be discarded")
		r128i abs32(r128i v) {
		return _mm_abs_epi32(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector math results must not be discarded")
		r128f32 sqrt(r128f32 v) {
		return _mm_sqrt_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector math results must not be discarded")
		r128f64 sqrt(r128f64 v) {
		return _mm_sqrt_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar math results must not be discarded")
		r128f32 sqrtScalar(r128f32 v) {
		return _mm_sqrt_ss(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar math results must not be discarded")
		r128f64 sqrtScalar(r128f64 base, r128f64 value) {
		return _mm_sqrt_sd(base, value);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector math results must not be discarded")
		r128f32 reciprocalSqrt(r128f32 v) {
		return _mm_rsqrt_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar math results must not be discarded")
		r128f32 reciprocalSqrtScalar(r128f32 v) {
		return _mm_rsqrt_ss(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector math results must not be discarded")
		r128f32 reciprocal(r128f32 v) {
		return _mm_rcp_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar math results must not be discarded")
		r128f32 reciprocalScalar(r128f32 v) {
		return _mm_rcp_ss(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r128f32 round(r128f32 v, int imm) {
		return _mm_round_ps(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r128f64 round(r128f64 v, int imm) {
		return _mm_round_pd(v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar rounding results must not be discarded")
		r128f32 roundScalar(r128f32 v, int imm) {
		return _mm_round_ss(v, v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar rounding results must not be discarded")
		r128f64 roundScalar(r128f64 v, int imm) {
		return _mm_round_sd(v, v, imm);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r128f32 ceil(r128f32 v) {
		return _mm_ceil_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r128f64 ceil(r128f64 v) {
		return _mm_ceil_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r128f32 floor(r128f32 v) {
		return _mm_floor_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector rounding results must not be discarded")
		r128f64 floor(r128f64 v) {
		return _mm_floor_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector packing results must not be discarded")
		r128i packSigned16(r128i a, r128i b) {
		return _mm_packs_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector packing results must not be discarded")
		r128i packSigned32(r128i a, r128i b) {
		return _mm_packs_epi32(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector packing results must not be discarded")
		r128i packUnsigned16(r128i a, r128i b) {
		return _mm_packus_epi16(a, b);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector packing results must not be discarded")
		r128i packUnsigned32(r128i a, r128i b) {
		return _mm_packus_epi32(a, b);
	}
	
	// float <-> double

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128f64 convert(r128f32 v) {
		return _mm_cvtps_pd(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128f32 convert(r128f64 v) {
		return _mm_cvtpd_ps(v);
	}

	// float -> int (rounded / truncated)

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i convertRound(r128f32 v) {
		return _mm_cvtps_epi32(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i convertTrunc(r128f32 v) {
		return _mm_cvttps_epi32(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i convertRound(r128f64 v) {
		return _mm_cvtpd_epi32(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i convertTrunc(r128f64 v) {
		return _mm_cvttpd_epi32(v);
	}

	// int -> float / double

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128f32 convert(r128i v) {
		return _mm_cvtepi32_ps(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128f64 convertDouble(r128i v) {
		return _mm_cvtepi32_pd(v);
	}

	// scalar extract / insert (integer)

	STL_FORCEINLINE
		STL_NODISCARD("Scalar conversion results must not be discarded")
		int toScalar32(r128i v) {
		return _mm_cvtsi128_si32(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar conversion results must not be discarded")
		r128i fromScalar32(int v) {
		return _mm_cvtsi32_si128(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar conversion results must not be discarded")
		long long toScalar64(r128i v) {
		return _mm_cvtsi128_si64(v);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Scalar conversion results must not be discarded")
		r128i fromScalar64(long long v) {
		return _mm_cvtsi64_si128(v);
	}

	// signed widening conversions

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widen8to16(r128i v) { return _mm_cvtepi8_epi16(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widen8to32(r128i v) { return _mm_cvtepi8_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widen8to64(r128i v) { return _mm_cvtepi8_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widen16to32(r128i v) { return _mm_cvtepi16_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widen16to64(r128i v) { return _mm_cvtepi16_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widen32to64(r128i v) { return _mm_cvtepi32_epi64(v); }

	// unsigned widening conversions

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widenU8to16(r128i v) { return _mm_cvtepu8_epi16(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widenU8to32(r128i v) { return _mm_cvtepu8_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widenU8to64(r128i v) { return _mm_cvtepu8_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widenU16to32(r128i v) { return _mm_cvtepu16_epi32(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widenU16to64(r128i v) { return _mm_cvtepu16_epi64(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector conversion results must not be discarded")
		r128i widenU32to64(r128i v) { return _mm_cvtepu32_epi64(v); }

	STL_FORCEINLINE r128f64 cast(r128f32 v) { return _mm_castps_pd(v); }
	STL_FORCEINLINE r128f32 cast(r128f64 v) { return _mm_castpd_ps(v); }

	STL_FORCEINLINE r128i  castInt(r128f32 v) { return _mm_castps_si128(v); }
	STL_FORCEINLINE r128f32 castFloat(r128i v) { return _mm_castsi128_ps(v); }

	STL_FORCEINLINE r128i  castInt(r128f64 v) { return _mm_castpd_si128(v); }
	STL_FORCEINLINE r128f64 castDouble(r128i v) { return _mm_castsi128_pd(v); }

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f32 fmadd(r128f32 a, r128f32 b, r128f32 c) {
		return _mm_fmadd_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f64 fmadd(r128f64 a, r128f64 b, r128f64 c) {
		return _mm_fmadd_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f32 fmsub(r128f32 a, r128f32 b, r128f32 c) {
		return _mm_fmsub_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f64 fmsub(r128f64 a, r128f64 b, r128f64 c) {
		return _mm_fmsub_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f32 fnmadd(r128f32 a, r128f32 b, r128f32 c) {
		return _mm_fnmadd_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f64 fnmadd(r128f64 a, r128f64 b, r128f64 c) {
		return _mm_fnmadd_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f32 fnmsub(r128f32 a, r128f32 b, r128f32 c) {
		return _mm_fnmsub_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f64 fnmsub(r128f64 a, r128f64 b, r128f64 c) {
		return _mm_fnmsub_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f32 fmaddsub(r128f32 a, r128f32 b, r128f32 c) {
		return _mm_fmaddsub_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f64 fmaddsub(r128f64 a, r128f64 b, r128f64 c) {
		return _mm_fmaddsub_pd(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f32 fmsubadd(r128f32 a, r128f32 b, r128f32 c) {
		return _mm_fmsubadd_ps(a, b, c);
	}

	STL_FORCEINLINE
		STL_NODISCARD("Vector FMA results must not be discarded")
		r128f64 fmsubadd(r128f64 a, r128f64 b, r128f64 c) {
		return _mm_fmsubadd_pd(a, b, c);
	}
}
#endif
#endif

