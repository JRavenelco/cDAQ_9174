// main.cu
#include <iostream>
__global__ void kernelTest() {
   // algo simple
}

int main() {
   kernelTest<<<1,1>>>();
   cudaDeviceSynchronize();
   std::cout << "Hola desde CUDA\n";
   return 0;
}
