#pragma once

#ifdef HEADER_EXPORTS  
#define HEADER_API __declspec(dllexport)   
#else  
#define HEADER_API __declspec(dllimport)   
#endif  


namespace Header
{
	// This class is exported from the MathLibrary.dll  
	class Functions
	{
	public:
		// Returns a + b  
		static HEADER_API double Add(double a, double b);

		// Returns a * b  
		static HEADER_API double Multiply(double a, double b);

		// Returns a + (a * b)  
		static HEADER_API double AddMultiply(double a, double b);
	};
}