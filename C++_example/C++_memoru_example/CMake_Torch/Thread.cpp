// CMake_Torch.cpp : Defines the entry point for the application.
//
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
int main() {
	// a pointer to integer
	int* ptr;
	//Allocates memory for an integer
	ptr = new int;
	//Assigns value to newly allocated int
	*ptr = 5;
	//Prints the value of int 
	std::cout << "\n\n\tint value=" << *ptr;
	//Prints memory location, where int is stored
	std::cout << "\n\n\tint stored at address=" << ptr << "\n\n";
	//Deallocate memory reserved for the int (to avoid memory leaks)
	delete ptr;
	return 0;
}
