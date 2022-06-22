/*#include <iostream>

constexpr int sqr(int x) {
    return x * x;
}
constexpr bool isPrime(int n) {
    if (n == 1) return false;
    for (int d = 2; sqr(d) <= n; ++d) {
        if (n % d == 0) return false;

    }
    return true;
}


int main() {
    std::cout << "Hello World!" << "\n";
    int g;
    std::cin >> g;
    std::cout << isPrime(g);



  
    return 0;
}*/



#include <iostream>
#include <string>
#include <thread>
#include <map>
#include <mutex>
#include <future>

std::mutex cout_mutex;
std::mutex cerr_mutex;

using dict_t = std::map<std::string, std::string>;

thread_local int thread_local_static_int = 0;
int static_int = 0;

std::string foo(dict_t& d)
{
    //    throw std::exception();
    thread_local_static_int = 0;
    static_int = 0;

    cout_mutex.lock();
    d["asas"] = "zxzxzx";
    cout_mutex.unlock();

    std::thread::id this_id = std::this_thread::get_id();

    cout_mutex.lock();
    std::cout << "foo " << this_id << std::endl;
    cout_mutex.unlock();

    cout_mutex.lock();
    std::cerr << "foo " << this_id << std::endl;
    cout_mutex.unlock();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    return std::string{ "ok" };
}

int main(int argc, char** argv)
{
    try {
        dict_t d, d2;
        foo(d);
        foo(std::ref(d));

        // казалось бы причём тут смарт поинтеры
        // старт предстакуемый
        std::thread t1(foo, std::ref(d));
        std::thread t2(foo, std::ref(d));

        // когда закончилось
        // чем закончилось
        t1.join();
        t2.join();

        auto r1 = std::async(std::launch::async,
            foo, std::ref(d));
        std::cout << r1.get() << std::endl;
        
       
    }
    catch (const std::exception&)
    {
        std::cerr << "oops" << std::endl;
    }
}