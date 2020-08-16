---
layout: post
title: SmallPRNG - A small header library to swap out random number generators
date: 2020-05-20
categories:
  - Personal Projects
---

With Monte Carlo methods it is well known that the quality of the results are sensitive on the quality of the random number generation. With some spectacular differences being seen in the literature [1](https://surface.syr.edu/cgi/viewcontent.cgi?article=1033&context=npac), [2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2992609/).

In fact a well known quote in pseudo-random number generation 
>"If all scientific papers whose results are in doubt because of [LCGs and related] were to disappear from library shelves, there would be a gap on each shelf about as big as your fist" - Numerical Recipes Press, et al.

Numerical Recipes underlines the results of LCG based prngs due to [Random numbers fall mainly in the Planes](https://www.pnas.org/content/pnas/61/1/25.full.pdf). Many common random library implementations rely on LCG prngs. For example, with excel we can see the random numbers falling in the planes.

![image](https://i.stack.imgur.com/JyKs8.png) 

Source:[https://stackoverflow.com/questions/38891165/is-excel-vbas-rnd-really-this-bad](https://stackoverflow.com/questions/38891165/is-excel-vbas-rnd-really-this-bad)

When generating points in 3D this effect becomes exacerbated

![image](https://upload.wikimedia.org/wikipedia/commons/a/a3/Lcg_3d.gif)

Source:[LCG Wikipedia](https://en.wikipedia.org/wiki/Linear_congruential_generator)

### Now the question is, will this effect my application? I am using a different prng, will this effect my results?

I wrote SmallPRNG, a small header library in C++ to answer this question. It lets the user swap in different prng implementations into a templated interface.

The user can swap in any deterministic prng into the prng construct and it will function as the randomness generator for the random number generator.

Here is an example of injecting the xorshift64 into a SmallPRNG(many common prngs are included in the library, including this one). First implement the specific prng.

```C++
_inline
uint64_t xorshift64(prng_state<2>& s) {
	uint64_t x = s.i64[0];
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	s.i64[0] = x;
	return x;
}

typedef prng<2, uint64_t, xorshift64> xorshift64_prng;

auto prng = xorshift_prng();

auto a = prng.rand() // returns a double in range (0, 1]
```
Currently you can generate random integers (with ranges), random floating points (with ranges), normaly and Poisson distributed samples. I intend on extending this to generate additional distributions so that it is more versatile.

You can find the [github repo here.](https://github.com/DKenefake/SmallPRNG) An depth walk thru on how to use this library is on the front page of the GitHub repo.