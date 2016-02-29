#ifndef __UTIL_H__
#define __UTIL_H__

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

namespace libdbscan {

template <typename TNum>
TNum euclidean_distance_nosse(size_t n, const TNum* x, const TNum* y){
	TNum result = 0.f;
	for (size_t i = 0; i < n; ++i) {
		const TNum num = x[i] - y[i];
		result += num * num;
	}
	return result;
}

template <typename TNum>
TNum euclidean_distance(size_t dim, const TNum* const x, const TNum* const y)
{
	return euclidean_distance_nosse<TNum>(dim, x, y);
}

#ifdef __SSE__

template <>
float euclidean_distance<float>(size_t n, const float* x, const float* y)
{
    // Specialization to do euclidean distance using single precision SSE1 SIMD
    // instructions.
    //
    // Note that the vectors must be of length > 3 for this to actually execute
    // any SIMD instructions.
    
	__m128 sum = _mm_setzero_ps();

    // parallel delta ^ 2 calcs stepping down the vectors in 128 bit blocks;
    // for single precision float SSE can do four elements at a time
	for (; n > 3; n -= 4) {
		const __m128 _x = _mm_loadu_ps(x);
		const __m128 _y = _mm_loadu_ps(y);
		const __m128 delta = _mm_sub_ps(_x, _y);
		const __m128 delta_squared = _mm_mul_ps(delta, delta);
		sum = _mm_add_ps(sum, delta_squared);
		x += 4;
		y += 4;
	}

    // dance to sum the 'sum' vector register into a single value - is there
    // a better way of doing this?
	const __m128 shuffle1 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1,0,3,2));
    // will contain [ sum[0] + sum[1], sum[1] + sum[0], 
    //                sum[2] + sum[3], sum[3] + sum[2] ]
	const __m128 sum1 = _mm_add_ps(sum, shuffle1);
	const __m128 shuffle2 = _mm_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2,3,0,1));
    // will contain [ (sum[0] + sum[1]) + (sum[2] + sum[3]), ... ]
	const __m128 sum2 = _mm_add_ps(sum1, shuffle2);

    // store lowest single precision value in distance i.e. our sum
    float distance;
	_mm_store_ss(&distance, sum2);

    // (we don't need _mm_empty here because the next instruction is an integer
    // comparison not a floating point instruction)
    //
    // deal with leftover values when the vector lengths aren't disible by 4
	if (n > 0) {
		distance += euclidean_distance_nosse(n, x, y);
    }

	return distance;
}

#endif

#ifdef __SSE2__

template <>
double euclidean_distance<double>(size_t n, const double* x, const double* y) 
{
    // Specialization to do euclidean distance using double precision SSE2 SIMD
    // instructions.
    //
    // Note that the vectors must be of length > 1 for this to actually execute
    // any SIMD instructions.

    __m128d sum = _mm_setzero_pd();

    // for double precision float SSE2 can do two elements at a time
    for (; n > 1; n -= 2) {
        const __m128d _x = _mm_loadu_pd(x);
        const __m128d _y = _mm_loadu_pd(y);
        const __m128d delta = _mm_sub_pd(_x, _y);
        const __m128d delta_squared = _mm_mul_pd(delta, delta);
        sum = _mm_add_pd(sum, delta_squared);
        x += 2;
        y += 2;
    }

	const __m128d shuffle1 = _mm_shuffle_pd(sum, sum, _MM_SHUFFLE2(0, 1));
    // will contain [ sum[0] + sum[1], sum[1] + sum[0] ]
	const __m128d sum1 = _mm_add_pd(sum, shuffle1);
    double distance;
	_mm_store_sd(&distance, sum1);

	if (n > 0) {
		distance += euclidean_distance_nosse(n, x, y);
    }

	return distance;
}

#endif

}

#endif
