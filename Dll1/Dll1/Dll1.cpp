// Dll1.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

#include "stdafx.h"  
#include "Header.h"  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
namespace Header
{
	double Functions::Add(double a, double b)
	{
		return a + b;
	}

	double Functions::Multiply(double a, double b)
	{
		return a * b;
	}

	double Functions::AddMultiply(double a, double b)
	{
		return a + (a * b);
	}
}

