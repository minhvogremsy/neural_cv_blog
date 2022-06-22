// CMake_Torch.cpp : Defines the entry point for the application.
//
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>



#include <thread>
#include <mutex>
#include <iostream>


//inline void MemoryBarrier() {
//        asm volatile("mfence" ::: "memory");
//}

void StoreBuffering(long int i) {
    std::atomic<int> x{ 0 }, y{ 0 };
    //std::atomic<int> y = 0;
    int r1 = 0;
    int r2 = 0;

    std::thread t1([&]() {
        x.store(1);
        r1 = y;
        });

    std::thread t2([&]() {
        y.store(1);
        r2 = x;
        });
    
    t1.join();
    t2.join();
    if (r1 == 0 && r2 == 0) {
        std::cout << "Iteretion #" << i << " :Brocen CPU" << std::endl;
        std::abort();
    }
    if (i > 100000) {
        std::cout << "Iteretion #" << i << " :NOT Brocen CPU" << std::endl;
        std::abort();
    }

}
int main()
{
    for (long int i = 0;; ++i) {
        StoreBuffering(i);
    }
    return 0;

    
}