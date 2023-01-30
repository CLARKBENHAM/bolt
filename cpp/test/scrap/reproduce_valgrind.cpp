// Check which VEX instruction Valgrind doesn't know

/* Compile and run from bolt/cpp
g++ -Og -march=native  -mavx -ffast-math -std=c++14 -ggdb -I/usr/local/include/eigen3 -Isrc/utils/ -o reproduce_valgrind test/scrap/reproduce_valgrind.cpp  
valgrind --tool=memcheck --track-origins=yes  ./reproduce_valgrind
// // Neither Errors if compile with -march=native, doesn't with -max2
g++ -Og -mavx2 -ffast-math -std=c++14 -ggdb -I/usr/local/include/eigen3 -Isrc/utils/ -o reproduce_valgrind test/scrap/reproduce_valgrind.cpp  
valgrind --tool=memcheck --track-origins=yes  ./reproduce_valgrind

*/
#include  <iostream>
#include "immintrin.h"

//This was the mre before compiling changed
int test_valgrind () {
  __m256 vmax = _mm256_set1_ps(1.1);
  auto out_scale = _mm256_castps256_ps128(vmax);
  std::cout << "starting:  if" << std::endl;
  if (out_scale[0] <= 0.0) {
      return 0; 
  }
  return out_scale[0];
}

// Never triggered valgrind error(!?)
//int test_valgrind () {
//  __m256 maxs = _mm256_set1_ps(9.1);
//  __m256 vmins = _mm256_set1_ps(0.1);
//  int ncodebooks=64;
//  float* out_offsets[64] = {};
//  float x = 0;
//  float y = 0;
//  float& out_offsets_sum = x;
//  float &out_scale = y;
//  void _compute_offsets_scale_from_mins_maxs(
//      const __m256* mins, const __m256* maxs, int ncodebooks,
//      float* out_offsets, float& out_offset_sum, float& out_scale);
//  
//}

int main() {
  test_valgrind();
}