---
layout: post
mathjax: true
title: Speed Differences between different RNGs
date: 2020-11-7
category:
  - Blog
---


Running Monti Carlo simulations rely on the fast generation of (and hopefully high quality) random numbers. Using my SmallPRNG library I investigated the speed of multiple different pseudo-random number generation algorithms for sampling 1 billion fp64 numbers on a 8600k running at 4.5 Ghz. 


|    Algorithm    |  Time (mSec) |
|:---------------:|:------------:|
|  Middle Square  |     2373     |
|    Xorshift32   |     2790     |
|    Xorshift64   |     1400     |
|   Xorshift128   |     2321     |
|   Xorshift128+  |     1367     |
|   XoShiro256**  |     5742     |
|   Knuth's LCG   |      926     |
| Improved Square |      229     |
|    Bob's PRNG   |      702     |
|     Salsa20     |     17638    |


We can see that Salsa20 is by far the slowest, this is do it being a cryptographically secure algorithm meant for excellent statistical quality and data security instead of speed. While the prng algorithms with higher "quality" usually take longer, the Improved Square algorithm is the fastest in the list while still having some strong statistical certificates (passes PractRand with 256Gb tested). However, in this case I suspect that the reason that Improved Square is so much faster is that the compiler is able to optimize away the prng itself. 


This can be seen in the dissassembly of the main benchmarking loop.

```nasm 
$LL433@main:

; 319  : 	s.i64[0]++;

	lea	rcx, QWORD PTR [rcx+4]
; File C:\Users\*********\RNG\Source.cpp

; 28   : 	for (uint64_t i = 0; i < 1000000000LL; i++)

	sub	rax, r13
	jne	SHORT $LL433@main

```


To run experiment on your machine compile and run the following
```cpp
#include"rng.h"
#include<iostream>
#include<chrono>
#include<functional>

//typdefs for the prngs
typedef prng<6, uint32_t, middle_square> mid_square;
typedef prng<1, uint32_t, xorshift32> xor32;
typedef prng<2, uint64_t, xorshift64> xor64;
typedef prng<4, uint32_t, xorshift128> xor128;
typedef prng<4, uint64_t, xorshift128plus> xor128_plus;
typedef prng<8, uint64_t, xoshiro256ss> xs_superstar;
typedef prng<2, uint64_t, fortran_lcg> knuth_lcg;
typedef prng<4, uint32_t, squares> improved_squares;
typedef prng<8, uint64_t, jsf> bob_prng;
typedef prng<33, uint32_t, salsa20> salsa;


//main benchmarking function
template<typename prng>
int benchmark() {
	
	auto start = std::chrono::high_resolution_clock::now();
	
	double sum = 0LL;

	auto my_prng = prng();

	for (uint64_t i = 0; i < 1000000000LL; i++)
		sum += my_prng.rand();

	auto end = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	return diff.count();
}


int main() {

	std::cout << "Algorithm        Time (mSec)" << std::endl;
	std::cout << "Middle Square    " << benchmark<mid_square>() << std::endl;
	std::cout << "Xorshift32       " << benchmark<xor32>() << std::endl;
	std::cout << "Xorshift64       " << benchmark<xor64>() << std::endl;
	std::cout << "Xorshift128      " << benchmark<xor128>() << std::endl;
	std::cout << "Xorshift128+     " << benchmark<xor128_plus>() << std::endl;
	std::cout << "Xoshiro256**     " << benchmark<xs_superstar>() << std::endl;
	std::cout << "Knuth's LCG      " << benchmark<knuth_lcg>() << std::endl;
	std::cout << "Improved Square  " << benchmark<improved_squares>() << std::endl;
	std::cout << "JSF              " << benchmark<bob_prng>() << std::endl;
	std::cout << "Salsa20          " << benchmark<salsa>() << std::endl;
	
	std::getchar();
	
	return 1;
}




```



