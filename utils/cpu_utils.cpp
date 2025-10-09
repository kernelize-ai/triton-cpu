#include <boost/fiber/all.hpp>

extern "C" void __attribute__((visibility("default"))) _cpu_barrier(void *b) {
  boost::fibers::barrier *barrier = (boost::fibers::barrier *)b;
  barrier->wait();
}
