// MathLibrary.cpp : Defines the exported functions for the DLL.
#include "pch.h" // use pch.h in Visual Studio 2019
#include <utility>
#include <limits.h>
#include "MathLibrary.h"



// Initialize a Fibonacci relation sequence
// such that F(0) = a, F(1) = b.
// This function must be called before any other function.
double
fprod(double x)
{
    if (x == 1. || x == 2.) return (x - 1.);
    return fprod(x - 1) + fprod(x - 2);

}