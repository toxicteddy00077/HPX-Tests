#include <hpx/hpx_main.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/local/execution.hpp>
#include <vector>
#include <cstddef>
#include <iostream>
#include <chrono>

using namespace std::chrono;

// tiled matrix multiplication
void tiled_matmul(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,std::size_t N, std::size_t blk_size) 
{
    // Initialize output matrix C to zero.
    std::fill(C.begin(), C.end(), 0.0);

    std::size_t blks=N/blk_size;

    // parallel loop over each individual block
    hpx::experimental::for_loop(hpx::execution::par, 0, blks, [&](std::size_t bi) {
        for (auto bj=0; bj<blks; bj++)
        {
            for (auto bk=0;bk<blks; bk++)
            {
                // individual block computation
                for (auto i=bi * blk_size; i<(bi+1)*blk_size; i++)
                {
                    for (auto j=bj*blk_size; j<(bj+1)*blk_size; j++)
                    {
                        float sum=0.0;
                        for (auto k=bk*blk_size; k<(bk+1)*blk_size; k++) sum+=A[i*N+k]*B[k*N+j];
                        C[i*N+j]+=sum;
                    }
                }
            }
        }
    });
}

int main(){
    std::size_t N = 1024; 
    std::size_t blk_size = 32;

    // Matrix initialization
    std::vector<float> A(N*N, 1.0); 
    std::vector<float> B(N*N, 2.0); 
    std::vector<float> C(N*N, 0.0); 

    high_resolution_clock::time_point start = high_resolution_clock::now();

    // Perform tiled matrix multiplication
    tiled_matmul(A, B, C, N, blk_size);

    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "Time taken for tiled matrix multiplication: " << duration << " ms" << std::endl;
    
    return 0;
}